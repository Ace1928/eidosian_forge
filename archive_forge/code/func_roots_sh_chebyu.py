import numpy as np
from numpy import (exp, inf, pi, sqrt, floor, sin, cos, around,
from scipy import linalg
from scipy.special import airy
from . import _specfun  # type: ignore
from . import _ufuncs
def roots_sh_chebyu(n, mu=False):
    """Gauss-Chebyshev (second kind, shifted) quadrature.

    Computes the sample points and weights for Gauss-Chebyshev
    quadrature. The sample points are the roots of the nth degree
    shifted Chebyshev polynomial of the second kind, :math:`U_n(x)`.
    These sample points and weights correctly integrate polynomials of
    degree :math:`2n - 1` or less over the interval :math:`[0, 1]`
    with weight function :math:`w(x) = \\sqrt{x - x^2}`. See 22.2.9 in
    [AS]_ for more details.

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
    x, w, m = roots_chebyu(n, True)
    x = (x + 1) / 2
    m_us = _ufuncs.beta(1.5, 1.5)
    w *= m_us / m
    if mu:
        return (x, w, m_us)
    else:
        return (x, w)