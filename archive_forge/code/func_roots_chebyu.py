import numpy as np
from numpy import (exp, inf, pi, sqrt, floor, sin, cos, around,
from scipy import linalg
from scipy.special import airy
from . import _specfun  # type: ignore
from . import _ufuncs
def roots_chebyu(n, mu=False):
    """Gauss-Chebyshev (second kind) quadrature.

    Computes the sample points and weights for Gauss-Chebyshev
    quadrature. The sample points are the roots of the nth degree
    Chebyshev polynomial of the second kind, :math:`U_n(x)`. These
    sample points and weights correctly integrate polynomials of
    degree :math:`2n - 1` or less over the interval :math:`[-1, 1]`
    with weight function :math:`w(x) = \\sqrt{1 - x^2}`. See 22.2.5 in
    [AS]_ for details.

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
    m = int(n)
    if n < 1 or n != m:
        raise ValueError('n must be a positive integer.')
    t = np.arange(m, 0, -1) * pi / (m + 1)
    x = np.cos(t)
    w = pi * np.sin(t) ** 2 / (m + 1)
    if mu:
        return (x, w, pi / 2)
    else:
        return (x, w)