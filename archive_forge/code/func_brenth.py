import warnings
from collections import namedtuple
import operator
from . import _zeros
from ._optimize import OptimizeResult, _call_callback_maybe_halt
import numpy as np
def brenth(f, a, b, args=(), xtol=_xtol, rtol=_rtol, maxiter=_iter, full_output=False, disp=True):
    """Find a root of a function in a bracketing interval using Brent's
    method with hyperbolic extrapolation.

    A variation on the classic Brent routine to find a root of the function f
    between the arguments a and b that uses hyperbolic extrapolation instead of
    inverse quadratic extrapolation. Bus & Dekker (1975) guarantee convergence
    for this method, claiming that the upper bound of function evaluations here
    is 4 or 5 times that of bisection.
    f(a) and f(b) cannot have the same signs. Generally, on a par with the
    brent routine, but not as heavily tested. It is a safe version of the
    secant method that uses hyperbolic extrapolation.
    The version here is by Chuck Harris, and implements Algorithm M of
    [BusAndDekker1975]_, where further details (convergence properties,
    additional remarks and such) can be found

    Parameters
    ----------
    f : function
        Python function returning a number. f must be continuous, and f(a) and
        f(b) must have opposite signs.
    a : scalar
        One end of the bracketing interval [a,b].
    b : scalar
        The other end of the bracketing interval [a,b].
    xtol : number, optional
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter must be positive. As with `brentq`, for nice
        functions the method will often satisfy the above condition
        with ``xtol/2`` and ``rtol/2``.
    rtol : number, optional
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter cannot be smaller than its default value of
        ``4*np.finfo(float).eps``. As with `brentq`, for nice functions
        the method will often satisfy the above condition with
        ``xtol/2`` and ``rtol/2``.
    maxiter : int, optional
        If convergence is not achieved in `maxiter` iterations, an error is
        raised. Must be >= 0.
    args : tuple, optional
        Containing extra arguments for the function `f`.
        `f` is called by ``apply(f, (x)+args)``.
    full_output : bool, optional
        If `full_output` is False, the root is returned. If `full_output` is
        True, the return value is ``(x, r)``, where `x` is the root, and `r` is
        a `RootResults` object.
    disp : bool, optional
        If True, raise RuntimeError if the algorithm didn't converge.
        Otherwise, the convergence status is recorded in any `RootResults`
        return object.

    Returns
    -------
    root : float
        Root of `f` between `a` and `b`.
    r : `RootResults` (present if ``full_output = True``)
        Object containing information about the convergence. In particular,
        ``r.converged`` is True if the routine converged.

    See Also
    --------
    fmin, fmin_powell, fmin_cg, fmin_bfgs, fmin_ncg : multivariate local optimizers
    leastsq : nonlinear least squares minimizer
    fmin_l_bfgs_b, fmin_tnc, fmin_cobyla : constrained multivariate optimizers
    basinhopping, differential_evolution, brute : global optimizers
    fminbound, brent, golden, bracket : local scalar minimizers
    fsolve : N-D root-finding
    brentq, brenth, ridder, bisect, newton : 1-D root-finding
    fixed_point : scalar fixed-point finder

    References
    ----------
    .. [BusAndDekker1975]
       Bus, J. C. P., Dekker, T. J.,
       "Two Efficient Algorithms with Guaranteed Convergence for Finding a Zero
       of a Function", ACM Transactions on Mathematical Software, Vol. 1, Issue
       4, Dec. 1975, pp. 330-345. Section 3: "Algorithm M".
       :doi:`10.1145/355656.355659`

    Examples
    --------
    >>> def f(x):
    ...     return (x**2 - 1)

    >>> from scipy import optimize

    >>> root = optimize.brenth(f, -2, 0)
    >>> root
    -1.0

    >>> root = optimize.brenth(f, 0, 2)
    >>> root
    1.0

    """
    if not isinstance(args, tuple):
        args = (args,)
    maxiter = operator.index(maxiter)
    if xtol <= 0:
        raise ValueError('xtol too small (%g <= 0)' % xtol)
    if rtol < _rtol:
        raise ValueError(f'rtol too small ({rtol:g} < {_rtol:g})')
    f = _wrap_nan_raise(f)
    r = _zeros._brenth(f, a, b, xtol, rtol, maxiter, args, full_output, disp)
    return results_c(full_output, r, 'brenth')