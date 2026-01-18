from collections import defaultdict
from nltk.classify.api import ClassifierI
from nltk.probability import FreqDist, MLEProbDist, entropy
@staticmethod
def best_binary_stump(feature_names, labeled_featuresets, feature_values, verbose=False):
    best_stump = DecisionTreeClassifier.leaf(labeled_featuresets)
    best_error = best_stump.error(labeled_featuresets)
    for fname in feature_names:
        for fval in feature_values[fname]:
            stump = DecisionTreeClassifier.binary_stump(fname, fval, labeled_featuresets)
            stump_error = stump.error(labeled_featuresets)
            if stump_error < best_error:
                best_error = stump_error
                best_stump = stump
    if verbose:
        if best_stump._decisions:
            descr = '{}={}'.format(best_stump._fname, list(best_stump._decisions.keys())[0])
        else:
            descr = '(default)'
        print('best stump for {:6d} toks uses {:20} err={:6.4f}'.format(len(labeled_featuresets), descr, best_error))
    return best_stump