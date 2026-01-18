import math as _math
from abc import ABCMeta, abstractmethod
from functools import reduce
@classmethod
def pmi(cls, *marginals):
    """Scores ngrams by pointwise mutual information, as in Manning and
        Schutze 5.4.
        """
    return _log2(marginals[NGRAM] * marginals[TOTAL] ** (cls._n - 1)) - _log2(_product(marginals[UNIGRAMS]))