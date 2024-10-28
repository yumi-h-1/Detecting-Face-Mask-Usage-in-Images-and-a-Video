# Detecting Face Mask Usage in Images and a Video
## Project Overview
This project focuses on developing a reliable model for detecting face masks in video. Four models were built and evaluated: two SVM (Support Vector Machine) models, an MLP (Multilayer Perceptron), and a pre-trained ResNet34 CNN (Convolutional Neural Network). Each model uses a distinct approach to feature extraction to compare their effectiveness in image classification tasks.

## Project Files
- **Data**: Cleaned data is stored in the `data/` and raw data is in the `data/raw` folder.
- **Notebooks**: The notebook for data cleaning is located in the `notebooks/` folder.
- **Scripts**: MATLAB scripts for training and testing models are located in the `scripts/` folder.
- **Results**: Trained Random Forest and Logistic Regression models are saved in the `results/models` folder. Visualisations, such as confusion matrix, can be found in the `results/figures` folder, with initial data analysis visualisations in the `results/initial_analysis/figures` folder.

## Methodology 
- **Data Preprocessing**: Filter the SBA Kaggle dataset (1987-2014) to California’s science and technology sector, resulting in 7,412 observations with 12 relevant variables. Set a binary target (0 = “paid-in-full,” 1 = “default”) with imbalanced classes. Remove non-predictive variables (e.g., company name) to focus on repayment indicators.
- **Feature Engineering**: Add SBA_proportion to represent the SBA loan’s share of the total loan amount, and retain RevLineCr (despite its 0.79 correlation with SBA_proportion) due to its relevance. Use Pearson correlation to assess feature relationships.
- **Modeling**: Add Gaussian noise to the training set to address target imbalance, build baseline models, and optimise via hyperparameter tuning.
- **Evaluation**: Assess models using accuracy and confusion matrices on the test set. Apply 5-fold cross-validation post-tuning.
***Random Forest (RF)***: Use ROC curves and AUC.
***Logistic Regression (LR)***: Use cross-entropy loss.
- **Final Model Evaluation**: Compare RF and LR on test set using confusion matrices, ROC, and AUC to select the best model for loan repayment prediction.

Models and Feature Extraction
Four models were created for this task:

SVM with SIFT: Used SIFT (Scale-Invariant Feature Transform) to extract key features from the images, which were then clustered via K-means (10x the number of labels) for training a Support Vector Machine (SVM).
SVM with HOG: Utilised HOG (Histogram of Oriented Gradients) to generate feature histograms from gradients, used to train another SVM.
MLP with HOG: HOG features were also used to train a Multilayer Perceptron (MLP) model, acting as a baseline neural network.
ResNet34: Employed a pre-trained ResNet34 model from PyTorch for convolutional feature extraction without additional feature engineering, as the CNN layers inherently capture essential patterns.
Preprocessing and Augmentation
Images were resized to 256x256 using interpolation. SVM and MLP datasets were split 80:20 for training and validation. ResNet34 images underwent additional augmentation (flipping, colour changes) to improve model generalisation, given class imbalances. Images were normalised by their calculated mean and standard deviation.

Training and Hyperparameter Optimisation
SVM and MLP: Grid search was applied to identify optimal hyperparameters. SVM tuning focused on regularisation C, gamma, and kernel type, while MLP underwent a two-stage search: the first round optimised hidden layers, activation, and optimiser; the second tuned regularisation alpha, learning rate, and momentum.
ResNet34: Fine-tuned with a lower learning rate (0.0001) and increased epochs (100) for better accuracy in face mask detection. Used the Adam optimiser to refine performance with smaller incremental updates.
Video Testing
For video testing, OpenCV's VideoCapture function detected faces, starting with a minimum size of 300x300, resizing faces to 224x224 for ResNet. The model applied bounding boxes and visual predictions on each detected face within the video frames.

## Key Findings
- **Model Comparison**: RF outperformed LR in accuracy and handling imbalanced data, while LR was more interpretable regarding feature significance. The RF model achieved a higher accuracy (93.5%) compared to LR (80.4%). Although LR had a slightly better recall (97.6% vs. RF’s 96.0%)—useful for identifying actual positive cases—RF had superior precision (95.5%) over LR (80.8%), making it more reliable for avoiding false positives. This reliability is critical in identifying companies likely to default. Additionally, RF performed better on the ROC curve, indicating effective handling of true positives and false positives. Testing times were low for both models: RF took slightly longer (0.121 seconds) than LR (0.005 seconds) on the current dataset size (7,000 observations). However, RF’s testing time is expected to increase more with larger datasets. Auto-tuning and cross-validation improved RF’s performance. Lasso regularisation (using MATLAB’s `lassoglm`) yielded a stable LR model by reducing non-informative features, although it showed relatively high cross-entropy error, limiting its effectiveness in this case
- **Challenges and Observations**: Both models achieved over 80% accuracy, though there remains a risk of overfitting due to the dataset’s imbalance and added noise. Careful noise addition is essential, as it can mislead models if complexity isn’t sufficient to generalise effectively. Future models should apply noise more cautiously.

## Used Datasets
- **Face Mask Video*: [How To Wear Face Mask The Right Way]([https://www.kaggle.com/datasets/mirbektoktogaraev/should-this-loan-be-approved-or-denied](https://youtu.be/W_9jLju5FuQ?feature=shared))
