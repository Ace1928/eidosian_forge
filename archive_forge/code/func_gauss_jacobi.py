from sympy.core import S, Dummy, pi
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.trigonometric import sin, cos
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.gamma_functions import gamma
from sympy.polys.orthopolys import (legendre_poly, laguerre_poly,
from sympy.polys.rootoftools import RootOf
def gauss_jacobi(n, alpha, beta, n_digits):
    """
    Computes the Gauss-Jacobi quadrature [1]_ points and weights.

    Explanation
    ===========

    The Gauss-Jacobi quadrature of the first kind approximates the integral:

    .. math::
        \\int_{-1}^1 (1-x)^\\alpha (1+x)^\\beta f(x)\\,dx \\approx
            \\sum_{i=1}^n w_i f(x_i)

    The nodes `x_i` of an order `n` quadrature rule are the roots of
    `P^{(\\alpha,\\beta)}_n` and the weights `w_i` are given by:

    .. math::
        w_i = -\\frac{2n+\\alpha+\\beta+2}{n+\\alpha+\\beta+1}
              \\frac{\\Gamma(n+\\alpha+1)\\Gamma(n+\\beta+1)}
              {\\Gamma(n+\\alpha+\\beta+1)(n+1)!}
              \\frac{2^{\\alpha+\\beta}}{P'_n(x_i)
              P^{(\\alpha,\\beta)}_{n+1}(x_i)}

    Parameters
    ==========

    n : the order of quadrature

    alpha : the first parameter of the Jacobi Polynomial, `\\alpha > -1`

    beta : the second parameter of the Jacobi Polynomial, `\\beta > -1`

    n_digits : number of significant digits of the points and weights to return

    Returns
    =======

    (x, w) : the ``x`` and ``w`` are lists of points and weights as Floats.
             The points `x_i` and weights `w_i` are returned as ``(x, w)``
             tuple of lists.

    Examples
    ========

    >>> from sympy import S
    >>> from sympy.integrals.quadrature import gauss_jacobi
    >>> x, w = gauss_jacobi(3, S.Half, -S.Half, 5)
    >>> x
    [-0.90097, -0.22252, 0.62349]
    >>> w
    [1.7063, 1.0973, 0.33795]

    >>> x, w = gauss_jacobi(6, 1, 1, 5)
    >>> x
    [-0.87174, -0.5917, -0.2093, 0.2093, 0.5917, 0.87174]
    >>> w
    [0.050584, 0.22169, 0.39439, 0.39439, 0.22169, 0.050584]

    See Also
    ========

    gauss_legendre, gauss_laguerre, gauss_hermite, gauss_gen_laguerre,
    gauss_chebyshev_t, gauss_chebyshev_u, gauss_lobatto

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gauss%E2%80%93Jacobi_quadrature
    .. [2] https://people.sc.fsu.edu/~jburkardt/cpp_src/jacobi_rule/jacobi_rule.html
    .. [3] https://people.sc.fsu.edu/~jburkardt/cpp_src/gegenbauer_rule/gegenbauer_rule.html
    """
    x = Dummy('x')
    p = jacobi_poly(n, alpha, beta, x, polys=True)
    pd = p.diff(x)
    pn = jacobi_poly(n + 1, alpha, beta, x, polys=True)
    xi = []
    w = []
    for r in p.real_roots():
        if isinstance(r, RootOf):
            r = r.eval_rational(S.One / 10 ** (n_digits + 2))
        xi.append(r.n(n_digits))
        w.append((-(2 * n + alpha + beta + 2) / (n + alpha + beta + S.One) * (gamma(n + alpha + 1) * gamma(n + beta + 1)) / (gamma(n + alpha + beta + S.One) * gamma(n + 2)) * 2 ** (alpha + beta) / (pd.subs(x, r) * pn.subs(x, r))).n(n_digits))
    return (xi, w)