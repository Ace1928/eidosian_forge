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
def show_most_informative_features(self, n=10, show='all'):
    """
        :param show: all, neg, or pos (for negative-only or positive-only)
        :type show: str
        :param n: The no. of top features
        :type n: int
        """
    fids = self.most_informative_features(None)
    if show == 'pos':
        fids = [fid for fid in fids if self._weights[fid] > 0]
    elif show == 'neg':
        fids = [fid for fid in fids if self._weights[fid] < 0]
    for fid in fids[:n]:
        print(f'{self._weights[fid]:8.3f} {self._encoding.describe(fid)}')