import logging
from itertools import groupby
from operator import itemgetter
from nltk.internals import deprecated
from nltk.metrics.distance import binary_distance
from nltk.probability import ConditionalFreqDist, FreqDist
def agr(self, cA, cB, i, data=None):
    """Agreement between two coders on a given item"""
    data = data or self.data
    k1 = next((x for x in data if x['coder'] in (cA, cB) and x['item'] == i))
    if k1['coder'] == cA:
        k2 = next((x for x in data if x['coder'] == cB and x['item'] == i))
    else:
        k2 = next((x for x in data if x['coder'] == cA and x['item'] == i))
    ret = 1.0 - float(self.distance(k1['labels'], k2['labels']))
    log.debug('Observed agreement between %s and %s on %s: %f', cA, cB, i, ret)
    log.debug('Distance between "%r" and "%r": %f', k1['labels'], k2['labels'], 1.0 - ret)
    return ret