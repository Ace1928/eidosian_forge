from itertools import product
import numpy as np
from numpy import (dot, diag, prod, logical_not, ravel, transpose,
from numpy.lib.scimath import sqrt as csqrt
from scipy.linalg import LinAlgError, bandwidth
from ._misc import norm
from ._basic import solve, inv
from ._decomp_svd import svd
from ._decomp_schur import schur, rsf2csf
from ._expm_frechet import expm_frechet, expm_cond
from ._matfuncs_sqrtm import sqrtm
from ._matfuncs_expm import pick_pade_structure, pade_UV_calc
from numpy import single  # noqa: F401
def rounded_sign(x):
    rx = np.real(x)
    if rx.dtype.char == 'f':
        c = 1000.0 * feps * amax(x)
    else:
        c = 1000.0 * eps * amax(x)
    return sign((absolute(rx) > c) * rx)