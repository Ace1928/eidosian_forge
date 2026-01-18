import logging
from itertools import groupby
from operator import itemgetter
from nltk.internals import deprecated
from nltk.metrics.distance import binary_distance
from nltk.probability import ConditionalFreqDist, FreqDist
def kappa_pairwise(self, cA, cB):
    """ """
    Ae = self.Ae_kappa(cA, cB)
    ret = (self.Ao(cA, cB) - Ae) / (1.0 - Ae)
    log.debug('Expected agreement between %s and %s: %f', cA, cB, Ae)
    return ret