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
def check_tolerance(ftol, xtol, gtol, method):

    def check(tol, name):
        if tol is None:
            tol = 0
        elif tol < EPS:
            warn(f'Setting `{name}` below the machine epsilon ({EPS:.2e}) effectively disables the corresponding termination condition.', stacklevel=3)
        return tol
    ftol = check(ftol, 'ftol')
    xtol = check(xtol, 'xtol')
    gtol = check(gtol, 'gtol')
    if method == 'lm' and (ftol < EPS or xtol < EPS or gtol < EPS):
        raise ValueError(f"All tolerances must be higher than machine epsilon ({EPS:.2e}) for method 'lm'.")
    elif ftol < EPS and xtol < EPS and (gtol < EPS):
        raise ValueError(f'At least one of the tolerances must be higher than machine epsilon ({EPS:.2e}).')
    return (ftol, xtol, gtol)