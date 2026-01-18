from collections import defaultdict
from nltk.classify.api import ClassifierI
from nltk.probability import FreqDist, MLEProbDist, entropy
@staticmethod
def binary_stump(feature_name, feature_value, labeled_featuresets):
    label = FreqDist((label for featureset, label in labeled_featuresets)).max()
    pos_fdist = FreqDist()
    neg_fdist = FreqDist()
    for featureset, label in labeled_featuresets:
        if featureset.get(feature_name) == feature_value:
            pos_fdist[label] += 1
        else:
            neg_fdist[label] += 1
    decisions = {}
    default = label
    if pos_fdist.N() > 0:
        decisions = {feature_value: DecisionTreeClassifier(pos_fdist.max())}
    if neg_fdist.N() > 0:
        default = DecisionTreeClassifier(neg_fdist.max())
    return DecisionTreeClassifier(label, feature_name, decisions, default)