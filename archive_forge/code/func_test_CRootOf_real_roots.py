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
def test_CRootOf_real_roots():
    assert Poly(x ** 5 + x + 1).real_roots() == [rootof(x ** 3 - x ** 2 + 1, 0)]
    assert Poly(x ** 5 + x + 1).real_roots(radicals=False) == [rootof(x ** 3 - x ** 2 + 1, 0)]
    p = Poly(-3 * x ** 4 - 10 * x ** 3 - 12 * x ** 2 - 6 * x - 1, x, domain='ZZ')
    assert CRootOf.real_roots(p) == [S(-1), S(-1), S(-1), S(-1) / 3]