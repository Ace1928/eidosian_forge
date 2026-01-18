import math as _math
from abc import ABCMeta, abstractmethod
from functools import reduce
@classmethod
def chi_sq(cls, n_ii, n_ix_xi_tuple, n_xx):
    """Scores bigrams using chi-square, i.e. phi-sq multiplied by the number
        of bigrams, as in Manning and Schutze 5.3.3.
        """
    n_ix, n_xi = n_ix_xi_tuple
    return n_xx * cls.phi_sq(n_ii, (n_ix, n_xi), n_xx)