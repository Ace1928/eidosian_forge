from collections import defaultdict
from nltk.classify.api import ClassifierI
from nltk.probability import DictionaryProbDist, ELEProbDist, FreqDist, sum_logs
def labelprob(l):
    return cpdist[l, fname].prob(fval)