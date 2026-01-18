from rdkit.ML.Data import SplitData
from rdkit.ML.KNN import DistFunctions
from rdkit.ML.KNN.KNNClassificationModel import KNNClassificationModel
from rdkit.ML.KNN.KNNRegressionModel import KNNRegressionModel
def makeRegressionModel(numNeigh, attrs, distFunc):
    return KNNRegressionModel(numNeigh, attrs, distFunc)