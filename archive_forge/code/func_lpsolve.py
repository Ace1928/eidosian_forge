import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import (get_blas_funcs, qr, solve, svd, qr_insert, lstsq)
from .iterative import _get_atol_rtol
from scipy.sparse.linalg._isolve.utils import make_system
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
def lpsolve(x):
    return x