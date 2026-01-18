from warnings import warn
import numpy as np
from numpy.linalg import norm
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator
from scipy.optimize import _minpack, OptimizeResult
from scipy.optimize._numdiff import approx_derivative, group_columns
from scipy.optimize._minimize import Bounds
from .trf import trf
from .dogbox import dogbox
from .common import EPS, in_bounds, make_strictly_feasible
def check_jac_sparsity(jac_sparsity, m, n):
    if jac_sparsity is None:
        return None
    if not issparse(jac_sparsity):
        jac_sparsity = np.atleast_2d(jac_sparsity)
    if jac_sparsity.shape != (m, n):
        raise ValueError('`jac_sparsity` has wrong shape.')
    return (jac_sparsity, group_columns(jac_sparsity))