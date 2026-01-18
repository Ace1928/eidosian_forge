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
def check_x_scale(x_scale, x0):
    if isinstance(x_scale, str) and x_scale == 'jac':
        return x_scale
    try:
        x_scale = np.asarray(x_scale, dtype=float)
        valid = np.all(np.isfinite(x_scale)) and np.all(x_scale > 0)
    except (ValueError, TypeError):
        valid = False
    if not valid:
        raise ValueError("`x_scale` must be 'jac' or array_like with positive numbers.")
    if x_scale.ndim == 0:
        x_scale = np.resize(x_scale, x0.shape)
    if x_scale.shape != x0.shape:
        raise ValueError('Inconsistent shapes between `x_scale` and `x0`.')
    return x_scale