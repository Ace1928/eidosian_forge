from rdkit.ML.Data import SplitData
from rdkit.ML.NaiveBayes.ClassificationModel import NaiveBayesClassifier
def makeNBClassificationModel(trainExamples, attrs, nPossibleValues, nQuantBounds, mEstimateVal=-1.0, useSigs=False, ensemble=None, useCMIM=0, **kwargs):
    if CMIM is not None and useCMIM > 0 and useSigs and (not ensemble):
        ensemble = CMIM.SelectFeatures(trainExamples, useCMIM, bvCol=1)
    if ensemble:
        attrs = ensemble
    model = NaiveBayesClassifier(attrs, nPossibleValues, nQuantBounds, mEstimateVal=mEstimateVal, useSigs=useSigs)
    model.SetTrainingExamples(trainExamples)
    model.trainModel()
    return model