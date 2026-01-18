from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Any, cast
import numpy as np
import numpy.typing as npt
import math
import warnings
from collections import namedtuple
from scipy.special import roots_legendre
from scipy.special import gammaln, logsumexp
from scipy._lib._util import _rng_spawn
from scipy._lib.deprecation import (_NoValue, _deprecate_positional_args,
@_deprecated('`scipy.integrate.romberg` is deprecated as of SciPy 1.12.0and will be removed in SciPy 1.15.0. Please use`scipy.integrate.quad` instead.')
def romberg(function, a, b, args=(), tol=1.48e-08, rtol=1.48e-08, show=False, divmax=10, vec_func=False):
    """
    Romberg integration of a callable function or method.

    .. deprecated:: 1.12.0

          This function is deprecated as of SciPy 1.12.0 and will be removed
          in SciPy 1.15.0. Please use `scipy.integrate.quad` instead.

    Returns the integral of `function` (a function of one variable)
    over the interval (`a`, `b`).

    If `show` is 1, the triangular array of the intermediate results
    will be printed. If `vec_func` is True (default is False), then
    `function` is assumed to support vector arguments.

    Parameters
    ----------
    function : callable
        Function to be integrated.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.

    Returns
    -------
    results : float
        Result of the integration.

    Other Parameters
    ----------------
    args : tuple, optional
        Extra arguments to pass to function. Each element of `args` will
        be passed as a single argument to `func`. Default is to pass no
        extra arguments.
    tol, rtol : float, optional
        The desired absolute and relative tolerances. Defaults are 1.48e-8.
    show : bool, optional
        Whether to print the results. Default is False.
    divmax : int, optional
        Maximum order of extrapolation. Default is 10.
    vec_func : bool, optional
        Whether `func` handles arrays as arguments (i.e., whether it is a
        "vector" function). Default is False.

    See Also
    --------
    fixed_quad : Fixed-order Gaussian quadrature.
    quad : Adaptive quadrature using QUADPACK.
    dblquad : Double integrals.
    tplquad : Triple integrals.
    romb : Integrators for sampled data.
    simpson : Integrators for sampled data.
    cumulative_trapezoid : Cumulative integration for sampled data.
    ode : ODE integrator.
    odeint : ODE integrator.

    References
    ----------
    .. [1] 'Romberg's method' https://en.wikipedia.org/wiki/Romberg%27s_method

    Examples
    --------
    Integrate a gaussian from 0 to 1 and compare to the error function.

    >>> from scipy import integrate
    >>> from scipy.special import erf
    >>> import numpy as np
    >>> gaussian = lambda x: 1/np.sqrt(np.pi) * np.exp(-x**2)
    >>> result = integrate.romberg(gaussian, 0, 1, show=True)
    Romberg integration of <function vfunc at ...> from [0, 1]

    ::

       Steps  StepSize  Results
           1  1.000000  0.385872
           2  0.500000  0.412631  0.421551
           4  0.250000  0.419184  0.421368  0.421356
           8  0.125000  0.420810  0.421352  0.421350  0.421350
          16  0.062500  0.421215  0.421350  0.421350  0.421350  0.421350
          32  0.031250  0.421317  0.421350  0.421350  0.421350  0.421350  0.421350

    The final result is 0.421350396475 after 33 function evaluations.

    >>> print("%g %g" % (2*result, erf(1)))
    0.842701 0.842701

    """
    if np.isinf(a) or np.isinf(b):
        raise ValueError('Romberg integration only available for finite limits.')
    vfunc = vectorize1(function, args, vec_func=vec_func)
    n = 1
    interval = [a, b]
    intrange = b - a
    ordsum = _difftrap(vfunc, interval, n)
    result = intrange * ordsum
    resmat = [[result]]
    err = np.inf
    last_row = resmat[0]
    for i in range(1, divmax + 1):
        n *= 2
        ordsum += _difftrap(vfunc, interval, n)
        row = [intrange * ordsum / n]
        for k in range(i):
            row.append(_romberg_diff(last_row[k], row[k], k + 1))
        result = row[i]
        lastresult = last_row[i - 1]
        if show:
            resmat.append(row)
        err = abs(result - lastresult)
        if err < tol or err < rtol * abs(result):
            break
        last_row = row
    else:
        warnings.warn('divmax (%d) exceeded. Latest difference = %e' % (divmax, err), AccuracyWarning, stacklevel=2)
    if show:
        _printresmat(vfunc, interval, resmat)
    return result