import logging
from itertools import groupby
from operator import itemgetter
from nltk.internals import deprecated
from nltk.metrics.distance import binary_distance
from nltk.probability import ConditionalFreqDist, FreqDist
def multi_kappa(self):
    """Davies and Fleiss 1982
        Averages over observed and expected agreements for each coder pair.

        """
    Ae = self._pairwise_average(self.Ae_kappa)
    return (self.avg_Ao() - Ae) / (1.0 - Ae)