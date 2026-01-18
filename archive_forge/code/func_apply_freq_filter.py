import itertools as _itertools
from nltk.metrics import (
from nltk.metrics.spearman import ranks_from_scores, spearman_correlation
from nltk.probability import FreqDist
from nltk.util import ngrams
def apply_freq_filter(self, min_freq):
    """Removes candidate ngrams which have frequency less than min_freq."""
    self._apply_filter(lambda ng, freq: freq < min_freq)