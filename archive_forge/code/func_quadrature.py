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
@_deprecated('`scipy.integrate.quadrature` is deprecated as of SciPy 1.12.0and will be removed in SciPy 1.15.0. Please use`scipy.integrate.quad` instead.')
def quadrature(func, a, b, args=(), tol=1.49e-08, rtol=1.49e-08, maxiter=50, vec_func=True, miniter=1):
    """
    Compute a definite integral using fixed-tolerance Gaussian quadrature.

    .. deprecated:: 1.12.0

          This function is deprecated as of SciPy 1.12.0 and will be removed
          in SciPy 1.15.0. Please use `scipy.integrate.quad` instead.

    Integrate `func` from `a` to `b` using Gaussian quadrature
    with absolute tolerance `tol`.

    Parameters
    ----------
    func : function
        A Python function or method to integrate.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.
    args : tuple, optional
        Extra arguments to pass to function.
    tol, rtol : float, optional
        Iteration stops when error between last two iterates is less than
        `tol` OR the relative change is less than `rtol`.
    maxiter : int, optional
        Maximum order of Gaussian quadrature.
    vec_func : bool, optional
        True or False if func handles arrays as arguments (is
        a "vector" function). Default is True.
    miniter : int, optional
        Minimum order of Gaussian quadrature.

    Returns
    -------
    val : float
        Gaussian quadrature approximation (within tolerance) to integral.
    err : float
        Difference between last two estimates of the integral.

    See Also
    --------
    romberg : adaptive Romberg quadrature
    fixed_quad : fixed-order Gaussian quadrature
    quad : adaptive quadrature using QUADPACK
    dblquad : double integrals
    tplquad : triple integrals
    romb : integrator for sampled data
    simpson : integrator for sampled data
    cumulative_trapezoid : cumulative integration for sampled data
    ode : ODE integrator
    odeint : ODE integrator

    Examples
    --------
    >>> from scipy import integrate
    >>> import numpy as np
    >>> f = lambda x: x**8
    >>> integrate.quadrature(f, 0.0, 1.0)
    (0.11111111111111106, 4.163336342344337e-17)
    >>> print(1/9.0)  # analytical result
    0.1111111111111111

    >>> integrate.quadrature(np.cos, 0.0, np.pi/2)
    (0.9999999999999536, 3.9611425250996035e-11)
    >>> np.sin(np.pi/2)-np.sin(0)  # analytical result
    1.0

    """
    if not isinstance(args, tuple):
        args = (args,)
    vfunc = vectorize1(func, args, vec_func=vec_func)
    val = np.inf
    err = np.inf
    maxiter = max(miniter + 1, maxiter)
    for n in range(miniter, maxiter + 1):
        newval = fixed_quad(vfunc, a, b, (), n)[0]
        err = abs(newval - val)
        val = newval
        if err < tol or err < rtol * abs(val):
            break
    else:
        warnings.warn('maxiter (%d) exceeded. Latest difference = %e' % (maxiter, err), AccuracyWarning, stacklevel=2)
    return (val, err)