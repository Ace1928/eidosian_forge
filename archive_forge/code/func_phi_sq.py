import math as _math
from abc import ABCMeta, abstractmethod
from functools import reduce
@classmethod
def phi_sq(cls, *marginals):
    """Scores bigrams using phi-square, the square of the Pearson correlation
        coefficient.
        """
    n_ii, n_io, n_oi, n_oo = cls._contingency(*marginals)
    return (n_ii * n_oo - n_io * n_oi) ** 2 / ((n_ii + n_io) * (n_ii + n_oi) * (n_io + n_oo) * (n_oi + n_oo))