import math as _math
from abc import ABCMeta, abstractmethod
from functools import reduce
@staticmethod
def raw_freq(*marginals):
    """Scores ngrams by their frequency"""
    return marginals[NGRAM] / marginals[TOTAL]