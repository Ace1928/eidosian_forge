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
def test_CRootOf___new__():
    assert rootof(x, 0) == 0
    assert rootof(x, -1) == 0
    assert rootof(x, S.Zero) == 0
    assert rootof(x - 1, 0) == 1
    assert rootof(x - 1, -1) == 1
    assert rootof(x + 1, 0) == -1
    assert rootof(x + 1, -1) == -1
    assert rootof(x ** 2 + 2 * x + 3, 0) == -1 - I * sqrt(2)
    assert rootof(x ** 2 + 2 * x + 3, 1) == -1 + I * sqrt(2)
    assert rootof(x ** 2 + 2 * x + 3, -1) == -1 + I * sqrt(2)
    assert rootof(x ** 2 + 2 * x + 3, -2) == -1 - I * sqrt(2)
    r = rootof(x ** 2 + 2 * x + 3, 0, radicals=False)
    assert isinstance(r, RootOf) is True
    r = rootof(x ** 2 + 2 * x + 3, 1, radicals=False)
    assert isinstance(r, RootOf) is True
    r = rootof(x ** 2 + 2 * x + 3, -1, radicals=False)
    assert isinstance(r, RootOf) is True
    r = rootof(x ** 2 + 2 * x + 3, -2, radicals=False)
    assert isinstance(r, RootOf) is True
    assert rootof((x - 1) * (x + 1), 0, radicals=False) == -1
    assert rootof((x - 1) * (x + 1), 1, radicals=False) == 1
    assert rootof((x - 1) * (x + 1), -1, radicals=False) == 1
    assert rootof((x - 1) * (x + 1), -2, radicals=False) == -1
    assert rootof((x - 1) * (x + 1), 0, radicals=True) == -1
    assert rootof((x - 1) * (x + 1), 1, radicals=True) == 1
    assert rootof((x - 1) * (x + 1), -1, radicals=True) == 1
    assert rootof((x - 1) * (x + 1), -2, radicals=True) == -1
    assert rootof((x - 1) * (x ** 3 + x + 3), 0) == rootof(x ** 3 + x + 3, 0)
    assert rootof((x - 1) * (x ** 3 + x + 3), 1) == 1
    assert rootof((x - 1) * (x ** 3 + x + 3), 2) == rootof(x ** 3 + x + 3, 1)
    assert rootof((x - 1) * (x ** 3 + x + 3), 3) == rootof(x ** 3 + x + 3, 2)
    assert rootof((x - 1) * (x ** 3 + x + 3), -1) == rootof(x ** 3 + x + 3, 2)
    assert rootof((x - 1) * (x ** 3 + x + 3), -2) == rootof(x ** 3 + x + 3, 1)
    assert rootof((x - 1) * (x ** 3 + x + 3), -3) == 1
    assert rootof((x - 1) * (x ** 3 + x + 3), -4) == rootof(x ** 3 + x + 3, 0)
    assert rootof(x ** 4 + 3 * x ** 3, 0) == -3
    assert rootof(x ** 4 + 3 * x ** 3, 1) == 0
    assert rootof(x ** 4 + 3 * x ** 3, 2) == 0
    assert rootof(x ** 4 + 3 * x ** 3, 3) == 0
    raises(GeneratorsNeeded, lambda: rootof(0, 0))
    raises(GeneratorsNeeded, lambda: rootof(1, 0))
    raises(PolynomialError, lambda: rootof(Poly(0, x), 0))
    raises(PolynomialError, lambda: rootof(Poly(1, x), 0))
    raises(PolynomialError, lambda: rootof(x - y, 0))
    raises(PolynomialError, lambda: rootof(exp(x), 0))
    raises(NotImplementedError, lambda: rootof(x ** 3 - x + sqrt(2), 0))
    raises(NotImplementedError, lambda: rootof(x ** 3 - x + I, 0))
    raises(IndexError, lambda: rootof(x ** 2 - 1, -4))
    raises(IndexError, lambda: rootof(x ** 2 - 1, -3))
    raises(IndexError, lambda: rootof(x ** 2 - 1, 2))
    raises(IndexError, lambda: rootof(x ** 2 - 1, 3))
    raises(ValueError, lambda: rootof(x ** 2 - 1, x))
    assert rootof(Poly(x - y, x), 0) == y
    assert rootof(Poly(x ** 2 - y, x), 0) == -sqrt(y)
    assert rootof(Poly(x ** 2 - y, x), 1) == sqrt(y)
    assert rootof(Poly(x ** 3 - y, x), 0) == y ** Rational(1, 3)
    assert rootof(y * x ** 3 + y * x + 2 * y, x, 0) == -1
    raises(NotImplementedError, lambda: rootof(x ** 3 + x + 2 * y, x, 0))
    assert rootof(x ** 3 + x + 1, 0).is_commutative is True