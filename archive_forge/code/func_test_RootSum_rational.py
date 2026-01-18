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
def test_RootSum_rational():
    assert RootSum(z ** 5 - z + 1, Lambda(z, z / (x - z))) == (4 * x - 5) / (x ** 5 - x + 1)
    f = 161 * z ** 3 + 115 * z ** 2 + 19 * z + 1
    g = Lambda(z, z * log(-3381 * z ** 4 / 4 - 3381 * z ** 3 / 4 - 625 * z ** 2 / 2 - z * Rational(125, 2) - 5 + exp(x)))
    assert RootSum(f, g).diff(x) == -((5 * exp(2 * x) - 6 * exp(x) + 4) * exp(x) / (exp(3 * x) - exp(2 * x) + 1)) / 7