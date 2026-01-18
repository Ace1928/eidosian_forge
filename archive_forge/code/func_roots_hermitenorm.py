import numpy as np
from numpy import (exp, inf, pi, sqrt, floor, sin, cos, around,
from scipy import linalg
from scipy.special import airy
from . import _specfun  # type: ignore
from . import _ufuncs
def roots_hermitenorm(n, mu=False):
    """Gauss-Hermite (statistician's) quadrature.

    Compute the sample points and weights for Gauss-Hermite
    quadrature. The sample points are the roots of the nth degree
    Hermite polynomial, :math:`He_n(x)`. These sample points and
    weights correctly integrate polynomials of degree :math:`2n - 1`
    or less over the interval :math:`[-\\infty, \\infty]` with weight
    function :math:`w(x) = e^{-x^2/2}`. See 22.2.15 in [AS]_ for more
    details.

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
    numpy.polynomial.hermite_e.hermegauss

    Notes
    -----
    For small n up to 150 a modified version of the Golub-Welsch
    algorithm is used. Nodes are computed from the eigenvalue
    problem and improved by one step of a Newton iteration.
    The weights are computed from the well-known analytical formula.

    For n larger than 150 an optimal asymptotic algorithm is used
    which computes nodes and weights in a numerical stable manner.
    The algorithm has linear runtime making computation for very
    large n (several thousand or more) feasible.

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    m = int(n)
    if n < 1 or n != m:
        raise ValueError('n must be a positive integer.')
    mu0 = np.sqrt(2.0 * np.pi)
    if n <= 150:

        def an_func(k):
            return 0.0 * k

        def bn_func(k):
            return np.sqrt(k)
        f = _ufuncs.eval_hermitenorm

        def df(n, x):
            return n * _ufuncs.eval_hermitenorm(n - 1, x)
        return _gen_roots_and_weights(m, mu0, an_func, bn_func, f, df, True, mu)
    else:
        nodes, weights = _roots_hermite_asy(m)
        nodes *= sqrt(2)
        weights *= sqrt(2)
        if mu:
            return (nodes, weights, mu0)
        else:
            return (nodes, weights)