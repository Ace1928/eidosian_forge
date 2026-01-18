import numpy as np
from numpy import (exp, inf, pi, sqrt, floor, sin, cos, around,
from scipy import linalg
from scipy.special import airy
from . import _specfun  # type: ignore
from . import _ufuncs
def roots_legendre(n, mu=False):
    """Gauss-Legendre quadrature.

    Compute the sample points and weights for Gauss-Legendre
    quadrature [GL]_. The sample points are the roots of the nth degree
    Legendre polynomial :math:`P_n(x)`. These sample points and
    weights correctly integrate polynomials of degree :math:`2n - 1`
    or less over the interval :math:`[-1, 1]` with weight function
    :math:`w(x) = 1`. See 2.2.10 in [AS]_ for more details.

    Parameters
    ----------
    n : int
        quadrature order
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.quadrature
    scipy.integrate.fixed_quad
    numpy.polynomial.legendre.leggauss

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.
    .. [GL] Gauss-Legendre quadrature, Wikipedia,
        https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import roots_legendre, eval_legendre
    >>> roots, weights = roots_legendre(9)

    ``roots`` holds the roots, and ``weights`` holds the weights for
    Gauss-Legendre quadrature.

    >>> roots
    array([-0.96816024, -0.83603111, -0.61337143, -0.32425342,  0.        ,
            0.32425342,  0.61337143,  0.83603111,  0.96816024])
    >>> weights
    array([0.08127439, 0.18064816, 0.2606107 , 0.31234708, 0.33023936,
           0.31234708, 0.2606107 , 0.18064816, 0.08127439])

    Verify that we have the roots by evaluating the degree 9 Legendre
    polynomial at ``roots``.  All the values are approximately zero:

    >>> eval_legendre(9, roots)
    array([-8.88178420e-16, -2.22044605e-16,  1.11022302e-16,  1.11022302e-16,
            0.00000000e+00, -5.55111512e-17, -1.94289029e-16,  1.38777878e-16,
           -8.32667268e-17])

    Here we'll show how the above values can be used to estimate the
    integral from 1 to 2 of f(t) = t + 1/t with Gauss-Legendre
    quadrature [GL]_.  First define the function and the integration
    limits.

    >>> def f(t):
    ...    return t + 1/t
    ...
    >>> a = 1
    >>> b = 2

    We'll use ``integral(f(t), t=a, t=b)`` to denote the definite integral
    of f from t=a to t=b.  The sample points in ``roots`` are from the
    interval [-1, 1], so we'll rewrite the integral with the simple change
    of variable::

        x = 2/(b - a) * t - (a + b)/(b - a)

    with inverse::

        t = (b - a)/2 * x + (a + 2)/2

    Then::

        integral(f(t), a, b) =
            (b - a)/2 * integral(f((b-a)/2*x + (a+b)/2), x=-1, x=1)

    We can approximate the latter integral with the values returned
    by `roots_legendre`.

    Map the roots computed above from [-1, 1] to [a, b].

    >>> t = (b - a)/2 * roots + (a + b)/2

    Approximate the integral as the weighted sum of the function values.

    >>> (b - a)/2 * f(t).dot(weights)
    2.1931471805599276

    Compare that to the exact result, which is 3/2 + log(2):

    >>> 1.5 + np.log(2)
    2.1931471805599454

    """
    m = int(n)
    if n < 1 or n != m:
        raise ValueError('n must be a positive integer.')
    mu0 = 2.0

    def an_func(k):
        return 0.0 * k

    def bn_func(k):
        return k * np.sqrt(1.0 / (4 * k * k - 1))
    f = _ufuncs.eval_legendre

    def df(n, x):
        return (-n * x * _ufuncs.eval_legendre(n, x) + n * _ufuncs.eval_legendre(n - 1, x)) / (1 - x ** 2)
    return _gen_roots_and_weights(m, mu0, an_func, bn_func, f, df, True, mu)