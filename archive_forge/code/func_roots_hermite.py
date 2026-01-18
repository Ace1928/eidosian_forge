import numpy as np
from numpy import (exp, inf, pi, sqrt, floor, sin, cos, around,
from scipy import linalg
from scipy.special import airy
from . import _specfun  # type: ignore
from . import _ufuncs
def roots_hermite(n, mu=False):
    """Gauss-Hermite (physicist's) quadrature.

    Compute the sample points and weights for Gauss-Hermite
    quadrature. The sample points are the roots of the nth degree
    Hermite polynomial, :math:`H_n(x)`. These sample points and
    weights correctly integrate polynomials of degree :math:`2n - 1`
    or less over the interval :math:`[-\\infty, \\infty]` with weight
    function :math:`w(x) = e^{-x^2}`. See 22.2.14 in [AS]_ for
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
    numpy.polynomial.hermite.hermgauss
    roots_hermitenorm

    Notes
    -----
    For small n up to 150 a modified version of the Golub-Welsch
    algorithm is used. Nodes are computed from the eigenvalue
    problem and improved by one step of a Newton iteration.
    The weights are computed from the well-known analytical formula.

    For n larger than 150 an optimal asymptotic algorithm is applied
    which computes nodes and weights in a numerically stable manner.
    The algorithm has linear runtime making computation for very
    large n (several thousand or more) feasible.

    References
    ----------
    .. [townsend.trogdon.olver-2014]
        Townsend, A. and Trogdon, T. and Olver, S. (2014)
        *Fast computation of Gauss quadrature nodes and
        weights on the whole real line*. :arXiv:`1410.5286`.
    .. [townsend.trogdon.olver-2015]
        Townsend, A. and Trogdon, T. and Olver, S. (2015)
        *Fast computation of Gauss quadrature nodes and
        weights on the whole real line*.
        IMA Journal of Numerical Analysis
        :doi:`10.1093/imanum/drv002`.
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    m = int(n)
    if n < 1 or n != m:
        raise ValueError('n must be a positive integer.')
    mu0 = np.sqrt(np.pi)
    if n <= 150:

        def an_func(k):
            return 0.0 * k

        def bn_func(k):
            return np.sqrt(k / 2.0)
        f = _ufuncs.eval_hermite

        def df(n, x):
            return 2.0 * n * _ufuncs.eval_hermite(n - 1, x)
        return _gen_roots_and_weights(m, mu0, an_func, bn_func, f, df, True, mu)
    else:
        nodes, weights = _roots_hermite_asy(m)
        if mu:
            return (nodes, weights, mu0)
        else:
            return (nodes, weights)