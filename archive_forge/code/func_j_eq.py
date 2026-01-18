import numpy as np
from ._hessian_update_strategy import BFGS
from ._differentiable_functions import (
from ._optimize import OptimizeWarning
from warnings import warn, catch_warnings, simplefilter, filterwarnings
from scipy.sparse import issparse
def j_eq(x):
    dy = jac(x)
    if issparse(dy):
        dy = dy.toarray()
    dy = np.atleast_2d(dy)
    return dy[i_eq, :]