from sympy.polys.polytools import Poly
import sympy.polys.rootoftools as rootoftools
from sympy.polys.rootoftools import (rootof, RootOf, CRootOf, RootSum,
from sympy.polys.polyerrors import (
from sympy.core.function import (Function, Lambda)
from sympy.core.numbers import (Float, I, Rational)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import tan
from sympy.integrals.integrals import Integral
from sympy.polys.orthopolys import legendre_poly
from sympy.solvers.solvers import solve
from sympy.testing.pytest import raises, slow
from sympy.core.expr import unchanged
from sympy.abc import a, b, x, y, z, r
def test__imag_count():
    from sympy.polys.rootoftools import _imag_count_of_factor

    def imag_count(p):
        return sum([_imag_count_of_factor(f) * m for f, m in p.factor_list()[1]])
    assert imag_count(Poly(x ** 6 + 10 * x ** 2 + 1)) == 2
    assert imag_count(Poly(x ** 2)) == 0
    assert imag_count(Poly([1] * 3 + [-1], x)) == 0
    assert imag_count(Poly(x ** 3 + 1)) == 0
    assert imag_count(Poly(x ** 2 + 1)) == 2
    assert imag_count(Poly(x ** 2 - 1)) == 0
    assert imag_count(Poly(x ** 4 - 1)) == 2
    assert imag_count(Poly(x ** 4 + 1)) == 0
    assert imag_count(Poly([1, 2, 3], x)) == 0
    assert imag_count(Poly(x ** 3 + x + 1)) == 0
    assert imag_count(Poly(x ** 4 + x + 1)) == 0

    def q(r1, r2, p):
        return Poly(((x - r1) * (x - r2)).subs(x, x ** p), x)
    assert imag_count(q(-1, -2, 2)) == 4
    assert imag_count(q(-1, 2, 2)) == 2
    assert imag_count(q(1, 2, 2)) == 0
    assert imag_count(q(1, 2, 4)) == 4
    assert imag_count(q(-1, 2, 4)) == 2
    assert imag_count(q(-1, -2, 4)) == 0