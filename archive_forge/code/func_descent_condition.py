import math
import warnings
import sys
import inspect
from numpy import (atleast_1d, eye, argmin, zeros, shape, squeeze,
import numpy as np
from scipy.linalg import cholesky, issymmetric, LinAlgError
from scipy.sparse.linalg import LinearOperator
from ._linesearch import (line_search_wolfe1, line_search_wolfe2,
from ._numdiff import approx_derivative
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy._lib._util import MapWrapper, check_random_state
from scipy.optimize._differentiable_functions import ScalarFunction, FD_METHODS
def descent_condition(alpha, xkp1, fp1, gfkp1):
    cached_step[:] = polak_ribiere_powell_step(alpha, gfkp1)
    alpha, xk, pk, gfk, gnorm = cached_step
    if gnorm <= gtol:
        return True
    return np.dot(pk, gfk) <= -sigma_3 * np.dot(gfk, gfk)