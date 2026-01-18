import numpy as np
from ._hessian_update_strategy import BFGS
from ._differentiable_functions import (
from ._optimize import OptimizeWarning
from warnings import warn, catch_warnings, simplefilter, filterwarnings
from scipy.sparse import issparse
def f_ineq(x):
    y = np.zeros(n_bound_below + n_bound_above)
    y_all = np.array(fun(x)).flatten()
    y[:n_bound_below] = y_all[i_bound_below] - lb[i_bound_below]
    y[n_bound_below:] = -(y_all[i_bound_above] - ub[i_bound_above])
    return y