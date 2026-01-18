from sympy.core import S, Dummy, pi
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.trigonometric import sin, cos
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.gamma_functions import gamma
from sympy.polys.orthopolys import (legendre_poly, laguerre_poly,
from sympy.polys.rootoftools import RootOf
def gauss_laguerre(n, n_digits):
    """
    Computes the Gauss-Laguerre quadrature [1]_ points and weights.

    Explanation
    ===========

    The Gauss-Laguerre quadrature approximates the integral:

    .. math::
        \\int_0^{\\infty} e^{-x} f(x)\\,dx \\approx \\sum_{i=1}^n w_i f(x_i)


    The nodes `x_i` of an order `n` quadrature rule are the roots of `L_n`
    and the weights `w_i` are given by:

    .. math::
        w_i = \\frac{x_i}{(n+1)^2 \\left(L_{n+1}(x_i)\\right)^2}

    Parameters
    ==========

    n :
        The order of quadrature.
    n_digits :
        Number of significant digits of the points and weights to return.

    Returns
    =======

    (x, w) : The ``x`` and ``w`` are lists of points and weights as Floats.
             The points `x_i` and weights `w_i` are returned as ``(x, w)``
             tuple of lists.

    Examples
    ========

    >>> from sympy.integrals.quadrature import gauss_laguerre
    >>> x, w = gauss_laguerre(3, 5)
    >>> x
    [0.41577, 2.2943, 6.2899]
    >>> w
    [0.71109, 0.27852, 0.010389]
    >>> x, w = gauss_laguerre(6, 5)
    >>> x
    [0.22285, 1.1889, 2.9927, 5.7751, 9.8375, 15.983]
    >>> w
    [0.45896, 0.417, 0.11337, 0.010399, 0.00026102, 8.9855e-7]

    See Also
    ========

    gauss_legendre, gauss_gen_laguerre, gauss_hermite, gauss_chebyshev_t, gauss_chebyshev_u, gauss_jacobi, gauss_lobatto

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gauss%E2%80%93Laguerre_quadrature
    .. [2] https://people.sc.fsu.edu/~jburkardt/cpp_src/laguerre_rule/laguerre_rule.html
    """
    x = Dummy('x')
    p = laguerre_poly(n, x, polys=True)
    p1 = laguerre_poly(n + 1, x, polys=True)
    xi = []
    w = []
    for r in p.real_roots():
        if isinstance(r, RootOf):
            r = r.eval_rational(S.One / 10 ** (n_digits + 2))
        xi.append(r.n(n_digits))
        w.append((r / ((n + 1) ** 2 * p1.subs(x, r) ** 2)).n(n_digits))
    return (xi, w)