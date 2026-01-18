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
def fun_wrapped(x):
    return np.atleast_1d(fun(x, *args, **kwargs))