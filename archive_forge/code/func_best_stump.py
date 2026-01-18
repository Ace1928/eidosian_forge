from collections import defaultdict
from nltk.classify.api import ClassifierI
from nltk.probability import FreqDist, MLEProbDist, entropy
@staticmethod
def best_stump(feature_names, labeled_featuresets, verbose=False):
    best_stump = DecisionTreeClassifier.leaf(labeled_featuresets)
    best_error = best_stump.error(labeled_featuresets)
    for fname in feature_names:
        stump = DecisionTreeClassifier.stump(fname, labeled_featuresets)
        stump_error = stump.error(labeled_featuresets)
        if stump_error < best_error:
            best_error = stump_error
            best_stump = stump
    if verbose:
        print('best stump for {:6d} toks uses {:20} err={:6.4f}'.format(len(labeled_featuresets), best_stump._fname, best_error))
    return best_stump