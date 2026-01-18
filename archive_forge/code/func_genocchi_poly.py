from sympy.polys.densearith import dup_mul_ground, dup_sub_ground, dup_quo_ground
from sympy.polys.densetools import dup_eval, dup_integrate
from sympy.polys.domains import ZZ, QQ
from sympy.polys.polytools import named_poly
from sympy.utilities import public
@public
def genocchi_poly(n, x=None, polys=False):
    """Generates the Genocchi polynomial `\\operatorname{G}_n(x)`.

    `\\operatorname{G}_n(x)` is twice the difference between the plain and
    central Bernoulli polynomials, so has degree `n-1`:

    .. math :: \\operatorname{G}_n(x) = 2 (\\operatorname{B}_n(x) -
            \\operatorname{B}_n^c(x))

    The factor of 2 in the definition endows `\\operatorname{G}_n(x)` with
    integer coefficients.

    Parameters
    ==========

    n : int
        Degree of the polynomial plus one.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.

    See Also
    ========

    sympy.functions.combinatorial.numbers.genocchi
    """
    return named_poly(n, dup_genocchi, ZZ, 'Genocchi polynomial', (x,), polys)