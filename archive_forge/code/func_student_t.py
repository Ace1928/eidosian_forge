import math as _math
from abc import ABCMeta, abstractmethod
from functools import reduce
@classmethod
def student_t(cls, *marginals):
    """Scores ngrams using Student's t test with independence hypothesis
        for unigrams, as in Manning and Schutze 5.3.1.
        """
    return (marginals[NGRAM] - _product(marginals[UNIGRAMS]) / marginals[TOTAL] ** (cls._n - 1)) / (marginals[NGRAM] + _SMALL) ** 0.5