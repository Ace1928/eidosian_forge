import numpy as np
import scipy.sparse as sps
from ._numdiff import approx_derivative, group_columns
from ._hessian_update_strategy import HessianUpdateStrategy
from scipy.sparse.linalg import LinearOperator
from scipy._lib._array_api import atleast_nd, array_namespace
def grad_wrapped(x):
    self.ngev += 1
    return np.atleast_1d(grad(np.copy(x), *args))