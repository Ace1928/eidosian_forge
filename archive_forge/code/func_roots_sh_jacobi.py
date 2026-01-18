import numpy as np
from numpy import (exp, inf, pi, sqrt, floor, sin, cos, around,
from scipy import linalg
from scipy.special import airy
from . import _specfun  # type: ignore
from . import _ufuncs
def roots_sh_jacobi(n, p1, q1, mu=False):
    """Gauss-Jacobi (shifted) quadrature.

    Compute the sample points and weights for Gauss-Jacobi (shifted)
    quadrature. The sample points are the roots of the nth degree
    shifted Jacobi polynomial, :math:`G^{p,q}_n(x)`. These sample
    points and weights correctly integrate polynomials of degree
    :math:`2n - 1` or less over the interval :math:`[0, 1]` with
    weight function :math:`w(x) = (1 - x)^{p-q} x^{q-1}`. See 22.2.2
    in [AS]_ for details.

    Parameters
    ----------
    n : int
        quadrature order
    p1 : float
        (p1 - q1) must be > -1
    q1 : float
        q1 must be > 0
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
    if p1 - q1 <= -1 or q1 <= 0:
        message = '(p - q) must be greater than -1, and q must be greater than 0.'
        raise ValueError(message)
    x, w, m = roots_jacobi(n, p1 - q1, q1 - 1, True)
    x = (x + 1) / 2
    scale = 2.0 ** p1
    w /= scale
    m /= scale
    if mu:
        return (x, w, m)
    else:
        return (x, w)