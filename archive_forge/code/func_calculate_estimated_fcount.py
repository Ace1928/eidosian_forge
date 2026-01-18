import os
import tempfile
from collections import defaultdict
from nltk.classify.api import ClassifierI
from nltk.classify.megam import call_megam, parse_megam_weights, write_megam_file
from nltk.classify.tadm import call_tadm, parse_tadm_weights, write_tadm_file
from nltk.classify.util import CutoffChecker, accuracy, log_likelihood
from nltk.data import gzip_open_unicode
from nltk.probability import DictionaryProbDist
from nltk.util import OrderedDict
def calculate_estimated_fcount(classifier, train_toks, encoding):
    fcount = numpy.zeros(encoding.length(), 'd')
    for tok, label in train_toks:
        pdist = classifier.prob_classify(tok)
        for label in pdist.samples():
            prob = pdist.prob(label)
            for fid, fval in encoding.encode(tok, label):
                fcount[fid] += prob * fval
    return fcount