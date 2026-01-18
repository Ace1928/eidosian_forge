from sympy.polys.partfrac import (
from sympy.core.expr import Expr
from sympy.core.function import Lambda
from sympy.core.numbers import (E, I, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.polys.polytools import (Poly, factor)
from sympy.polys.rationaltools import together
from sympy.polys.rootoftools import RootSum
from sympy.testing.pytest import raises, XFAIL
from sympy.abc import x, y, a, b, c
def test_apart_undetermined_coeffs():
    p = Poly(2 * x - 3)
    q = Poly(x ** 9 - x ** 8 - x ** 6 + x ** 5 - 2 * x ** 2 + 3 * x - 1)
    r = (-x ** 7 - x ** 6 - x ** 5 + 4) / (x ** 8 - x ** 5 - 2 * x + 1) + 1 / (x - 1)
    assert apart_undetermined_coeffs(p, q) == r
    p = Poly(1, x, domain='ZZ[a,b]')
    q = Poly((x + a) * (x + b), x, domain='ZZ[a,b]')
    r = 1 / ((a - b) * (b + x)) - 1 / ((a - b) * (a + x))
    assert apart_undetermined_coeffs(p, q) == r