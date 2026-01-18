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
def most_informative_features(self, n=10):
    """
        Generates the ranked list of informative features from most to least.
        """
    if hasattr(self, '_most_informative_features'):
        return self._most_informative_features[:n]
    else:
        self._most_informative_features = sorted(list(range(len(self._weights))), key=lambda fid: abs(self._weights[fid]), reverse=True)
        return self._most_informative_features[:n]