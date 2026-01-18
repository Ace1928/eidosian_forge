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
def rosen_hess(x):
    """
    The Hessian matrix of the Rosenbrock function.

    Parameters
    ----------
    x : array_like
        1-D array of points at which the Hessian matrix is to be computed.

    Returns
    -------
    rosen_hess : ndarray
        The Hessian matrix of the Rosenbrock function at `x`.

    See Also
    --------
    rosen, rosen_der, rosen_hess_prod

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.optimize import rosen_hess
    >>> X = 0.1 * np.arange(4)
    >>> rosen_hess(X)
    array([[-38.,   0.,   0.,   0.],
           [  0., 134., -40.,   0.],
           [  0., -40., 130., -80.],
           [  0.,   0., -80., 200.]])

    """
    x = atleast_1d(x)
    H = np.diag(-400 * x[:-1], 1) - np.diag(400 * x[:-1], -1)
    diagonal = np.zeros(len(x), dtype=x.dtype)
    diagonal[0] = 1200 * x[0] ** 2 - 400 * x[1] + 2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200 * x[1:-1] ** 2 - 400 * x[2:]
    H = H + np.diag(diagonal)
    return H