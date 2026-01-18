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
def test_CRootOf___eval_Eq__():
    f = Function('f')
    eq = x ** 3 + x + 3
    r = rootof(eq, 2)
    r1 = rootof(eq, 1)
    assert Eq(r, r1) is S.false
    assert Eq(r, r) is S.true
    assert unchanged(Eq, r, x)
    assert Eq(r, 0) is S.false
    assert Eq(r, S.Infinity) is S.false
    assert Eq(r, I) is S.false
    assert unchanged(Eq, r, f(0))
    sol = solve(eq)
    for s in sol:
        if s.is_real:
            assert Eq(r, s) is S.false
    r = rootof(eq, 0)
    for s in sol:
        if s.is_real:
            assert Eq(r, s) is S.true
    eq = x ** 3 + x + 1
    sol = solve(eq)
    assert [Eq(rootof(eq, i), j) for i in range(3) for j in sol].count(True) == 3
    assert Eq(rootof(eq, 0), 1 + S.ImaginaryUnit) == False