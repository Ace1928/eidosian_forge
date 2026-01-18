import os
import numpy as np
from numpy import (asarray, real, imag, conj, zeros, ndarray, concatenate,
from scipy.sparse import coo_matrix, issparse
@property
def has_symmetry(self):
    return self._symmetry in (self.SYMMETRY_SYMMETRIC, self.SYMMETRY_SKEW_SYMMETRIC, self.SYMMETRY_HERMITIAN)