from sympy.polys.densearith import dup_mul_ground, dup_sub_ground, dup_quo_ground
from sympy.polys.densetools import dup_eval, dup_integrate
from sympy.polys.domains import ZZ, QQ
from sympy.polys.polytools import named_poly
from sympy.utilities import public
@public
def euler_poly(n, x=None, polys=False):
    """Generates the Euler polynomial `\\operatorname{E}_n(x)`.

    These are scaled and reindexed versions of the Genocchi polynomials:

    .. math :: \\operatorname{E}_n(x) = -\\frac{\\operatorname{G}_{n+1}(x)}{n+1}

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.

    See Also
    ========

    sympy.functions.combinatorial.numbers.euler
    """
    return named_poly(n, dup_euler, QQ, 'Euler polynomial', (x,), polys)