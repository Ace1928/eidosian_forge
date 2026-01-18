from sympy.core import Add, Expr, Mul, S, sympify
from sympy.core.function import _mexpand, count_ops, expand_mul
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy
from sympy.functions import root, sign, sqrt
from sympy.polys import Poly, PolynomialError
def sqrt_biquadratic_denest(expr, a, b, r, d2):
    """denest expr = sqrt(a + b*sqrt(r))
    where a, b, r are linear combinations of square roots of
    positive rationals on the rationals (SQRR) and r > 0, b != 0,
    d2 = a**2 - b**2*r > 0

    If it cannot denest it returns None.

    Explanation
    ===========

    Search for a solution A of type SQRR of the biquadratic equation
    4*A**4 - 4*a*A**2 + b**2*r = 0                               (1)
    sqd = sqrt(a**2 - b**2*r)
    Choosing the sqrt to be positive, the possible solutions are
    A = sqrt(a/2 +/- sqd/2)
    Since a, b, r are SQRR, then a**2 - b**2*r is a SQRR,
    so if sqd can be denested, it is done by
    _sqrtdenest_rec, and the result is a SQRR.
    Similarly for A.
    Examples of solutions (in both cases a and sqd are positive):

      Example of expr with solution sqrt(a/2 + sqd/2) but not
      solution sqrt(a/2 - sqd/2):
      expr = sqrt(-sqrt(15) - sqrt(2)*sqrt(-sqrt(5) + 5) - sqrt(3) + 8)
      a = -sqrt(15) - sqrt(3) + 8; sqd = -2*sqrt(5) - 2 + 4*sqrt(3)

      Example of expr with solution sqrt(a/2 - sqd/2) but not
      solution sqrt(a/2 + sqd/2):
      w = 2 + r2 + r3 + (1 + r3)*sqrt(2 + r2 + 5*r3)
      expr = sqrt((w**2).expand())
      a = 4*sqrt(6) + 8*sqrt(2) + 47 + 28*sqrt(3)
      sqd = 29 + 20*sqrt(3)

    Define B = b/2*A; eq.(1) implies a = A**2 + B**2*r; then
    expr**2 = a + b*sqrt(r) = (A + B*sqrt(r))**2

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.simplify.sqrtdenest import _sqrt_match, sqrt_biquadratic_denest
    >>> z = sqrt((2*sqrt(2) + 4)*sqrt(2 + sqrt(2)) + 5*sqrt(2) + 8)
    >>> a, b, r = _sqrt_match(z**2)
    >>> d2 = a**2 - b**2*r
    >>> sqrt_biquadratic_denest(z, a, b, r, d2)
    sqrt(2) + sqrt(sqrt(2) + 2) + 2
    """
    from sympy.simplify.radsimp import radsimp, rad_rationalize
    if r <= 0 or d2 < 0 or (not b) or (sqrt_depth(expr.base) < 2):
        return None
    for x in (a, b, r):
        for y in x.args:
            y2 = y ** 2
            if not y2.is_Integer or not y2.is_positive:
                return None
    sqd = _mexpand(sqrtdenest(sqrt(radsimp(d2))))
    if sqrt_depth(sqd) > 1:
        return None
    x1, x2 = [a / 2 + sqd / 2, a / 2 - sqd / 2]
    for x in (x1, x2):
        A = sqrtdenest(sqrt(x))
        if sqrt_depth(A) > 1:
            continue
        Bn, Bd = rad_rationalize(b, _mexpand(2 * A))
        B = Bn / Bd
        z = A + B * sqrt(r)
        if z < 0:
            z = -z
        return _mexpand(z)
    return None