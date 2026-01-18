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
def fmin_powell(func, x0, args=(), xtol=0.0001, ftol=0.0001, maxiter=None, maxfun=None, full_output=0, disp=1, retall=0, callback=None, direc=None):
    """
    Minimize a function using modified Powell's method.

    This method only uses function values, not derivatives.

    Parameters
    ----------
    func : callable f(x,*args)
        Objective function to be minimized.
    x0 : ndarray
        Initial guess.
    args : tuple, optional
        Extra arguments passed to func.
    xtol : float, optional
        Line-search error tolerance.
    ftol : float, optional
        Relative error in ``func(xopt)`` acceptable for convergence.
    maxiter : int, optional
        Maximum number of iterations to perform.
    maxfun : int, optional
        Maximum number of function evaluations to make.
    full_output : bool, optional
        If True, ``fopt``, ``xi``, ``direc``, ``iter``, ``funcalls``, and
        ``warnflag`` are returned.
    disp : bool, optional
        If True, print convergence messages.
    retall : bool, optional
        If True, return a list of the solution at each iteration.
    callback : callable, optional
        An optional user-supplied function, called after each
        iteration.  Called as ``callback(xk)``, where ``xk`` is the
        current parameter vector.
    direc : ndarray, optional
        Initial fitting step and parameter order set as an (N, N) array, where N
        is the number of fitting parameters in `x0`. Defaults to step size 1.0
        fitting all parameters simultaneously (``np.eye((N, N))``). To
        prevent initial consideration of values in a step or to change initial
        step size, set to 0 or desired step size in the Jth position in the Mth
        block, where J is the position in `x0` and M is the desired evaluation
        step, with steps being evaluated in index order. Step size and ordering
        will change freely as minimization proceeds.

    Returns
    -------
    xopt : ndarray
        Parameter which minimizes `func`.
    fopt : number
        Value of function at minimum: ``fopt = func(xopt)``.
    direc : ndarray
        Current direction set.
    iter : int
        Number of iterations.
    funcalls : int
        Number of function calls made.
    warnflag : int
        Integer warning flag:
            1 : Maximum number of function evaluations.
            2 : Maximum number of iterations.
            3 : NaN result encountered.
            4 : The result is out of the provided bounds.
    allvecs : list
        List of solutions at each iteration.

    See also
    --------
    minimize: Interface to unconstrained minimization algorithms for
        multivariate functions. See the 'Powell' method in particular.

    Notes
    -----
    Uses a modification of Powell's method to find the minimum of
    a function of N variables. Powell's method is a conjugate
    direction method.

    The algorithm has two loops. The outer loop merely iterates over the inner
    loop. The inner loop minimizes over each current direction in the direction
    set. At the end of the inner loop, if certain conditions are met, the
    direction that gave the largest decrease is dropped and replaced with the
    difference between the current estimated x and the estimated x from the
    beginning of the inner-loop.

    The technical conditions for replacing the direction of greatest
    increase amount to checking that

    1. No further gain can be made along the direction of greatest increase
       from that iteration.
    2. The direction of greatest increase accounted for a large sufficient
       fraction of the decrease in the function value from that iteration of
       the inner loop.

    References
    ----------
    Powell M.J.D. (1964) An efficient method for finding the minimum of a
    function of several variables without calculating derivatives,
    Computer Journal, 7 (2):155-162.

    Press W., Teukolsky S.A., Vetterling W.T., and Flannery B.P.:
    Numerical Recipes (any edition), Cambridge University Press

    Examples
    --------
    >>> def f(x):
    ...     return x**2

    >>> from scipy import optimize

    >>> minimum = optimize.fmin_powell(f, -1)
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 2
             Function evaluations: 16
    >>> minimum
    array(0.0)

    """
    opts = {'xtol': xtol, 'ftol': ftol, 'maxiter': maxiter, 'maxfev': maxfun, 'disp': disp, 'direc': direc, 'return_all': retall}
    callback = _wrap_callback(callback)
    res = _minimize_powell(func, x0, args, callback=callback, **opts)
    if full_output:
        retlist = (res['x'], res['fun'], res['direc'], res['nit'], res['nfev'], res['status'])
        if retall:
            retlist += (res['allvecs'],)
        return retlist
    elif retall:
        return (res['x'], res['allvecs'])
    else:
        return res['x']