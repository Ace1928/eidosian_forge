from collections import defaultdict
from nltk.classify.api import ClassifierI
from nltk.probability import FreqDist, MLEProbDist, entropy
@staticmethod
def stump(feature_name, labeled_featuresets):
    label = FreqDist((label for featureset, label in labeled_featuresets)).max()
    freqs = defaultdict(FreqDist)
    for featureset, label in labeled_featuresets:
        feature_value = featureset.get(feature_name)
        freqs[feature_value][label] += 1
    decisions = {val: DecisionTreeClassifier(freqs[val].max()) for val in freqs}
    return DecisionTreeClassifier(label, feature_name, decisions)