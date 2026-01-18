import numpy as np
from ._hessian_update_strategy import BFGS
from ._differentiable_functions import (
from ._optimize import OptimizeWarning
from warnings import warn, catch_warnings, simplefilter, filterwarnings
from scipy.sparse import issparse
def f_eq(x):
    y = np.array(fun(x)).flatten()
    return y[i_eq] - lb[i_eq]