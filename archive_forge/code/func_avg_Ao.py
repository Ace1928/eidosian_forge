import logging
from itertools import groupby
from operator import itemgetter
from nltk.internals import deprecated
from nltk.metrics.distance import binary_distance
from nltk.probability import ConditionalFreqDist, FreqDist
def avg_Ao(self):
    """Average observed agreement across all coders and items."""
    ret = self._pairwise_average(self.Ao)
    log.debug('Average observed agreement: %f', ret)
    return ret