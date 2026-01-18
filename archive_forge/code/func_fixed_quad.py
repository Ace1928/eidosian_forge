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
def fixed_quad(func, a, b, args=(), n=5):
    """
    Compute a definite integral using fixed-order Gaussian quadrature.

    Integrate `func` from `a` to `b` using Gaussian quadrature of
    order `n`.

    Parameters
    ----------
    func : callable
        A Python function or method to integrate (must accept vector inputs).
        If integrating a vector-valued function, the returned array must have
        shape ``(..., len(x))``.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.
    args : tuple, optional
        Extra arguments to pass to function, if any.
    n : int, optional
        Order of quadrature integration. Default is 5.

    Returns
    -------
    val : float
        Gaussian quadrature approximation to the integral
    none : None
        Statically returned value of None

    See Also
    --------
    quad : adaptive quadrature using QUADPACK
    dblquad : double integrals
    tplquad : triple integrals
    romberg : adaptive Romberg quadrature
    quadrature : adaptive Gaussian quadrature
    romb : integrators for sampled data
    simpson : integrators for sampled data
    cumulative_trapezoid : cumulative integration for sampled data
    ode : ODE integrator
    odeint : ODE integrator

    Examples
    --------
    >>> from scipy import integrate
    >>> import numpy as np
    >>> f = lambda x: x**8
    >>> integrate.fixed_quad(f, 0.0, 1.0, n=4)
    (0.1110884353741496, None)
    >>> integrate.fixed_quad(f, 0.0, 1.0, n=5)
    (0.11111111111111102, None)
    >>> print(1/9.0)  # analytical result
    0.1111111111111111

    >>> integrate.fixed_quad(np.cos, 0.0, np.pi/2, n=4)
    (0.9999999771971152, None)
    >>> integrate.fixed_quad(np.cos, 0.0, np.pi/2, n=5)
    (1.000000000039565, None)
    >>> np.sin(np.pi/2)-np.sin(0)  # analytical result
    1.0

    """
    x, w = _cached_roots_legendre(n)
    x = np.real(x)
    if np.isinf(a) or np.isinf(b):
        raise ValueError('Gaussian quadrature is only available for finite limits.')
    y = (b - a) * (x + 1) / 2.0 + a
    return ((b - a) / 2.0 * np.sum(w * func(y, *args), axis=-1), None)