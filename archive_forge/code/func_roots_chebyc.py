import numpy as np
from numpy import (exp, inf, pi, sqrt, floor, sin, cos, around,
from scipy import linalg
from scipy.special import airy
from . import _specfun  # type: ignore
from . import _ufuncs
def roots_chebyc(n, mu=False):
    """Gauss-Chebyshev (first kind) quadrature.

    Compute the sample points and weights for Gauss-Chebyshev
    quadrature. The sample points are the roots of the nth degree
    Chebyshev polynomial of the first kind, :math:`C_n(x)`. These
    sample points and weights correctly integrate polynomials of
    degree :math:`2n - 1` or less over the interval :math:`[-2, 2]`
    with weight function :math:`w(x) = 1 / \\sqrt{1 - (x/2)^2}`. See
    22.2.6 in [AS]_ for more details.

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

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    x, w, m = roots_chebyt(n, True)
    x *= 2
    w *= 2
    m *= 2
    if mu:
        return (x, w, m)
    else:
        return (x, w)