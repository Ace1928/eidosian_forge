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
def test_RootSum_independent():
    f = (x ** 3 - a) ** 2 * (x ** 4 - b) ** 3
    g = Lambda(x, 5 * tan(x) + 7)
    h = Lambda(x, tan(x))
    r0 = RootSum(x ** 3 - a, h, x)
    r1 = RootSum(x ** 4 - b, h, x)
    assert RootSum(f, g, x).as_ordered_terms() == [10 * r0, 15 * r1, 126]