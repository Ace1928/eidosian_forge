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
def test_CRootOf___eq__():
    assert (rootof(x ** 3 + x + 3, 0) == rootof(x ** 3 + x + 3, 0)) is True
    assert (rootof(x ** 3 + x + 3, 0) == rootof(x ** 3 + x + 3, 1)) is False
    assert (rootof(x ** 3 + x + 3, 1) == rootof(x ** 3 + x + 3, 1)) is True
    assert (rootof(x ** 3 + x + 3, 1) == rootof(x ** 3 + x + 3, 2)) is False
    assert (rootof(x ** 3 + x + 3, 2) == rootof(x ** 3 + x + 3, 2)) is True
    assert (rootof(x ** 3 + x + 3, 0) == rootof(y ** 3 + y + 3, 0)) is True
    assert (rootof(x ** 3 + x + 3, 0) == rootof(y ** 3 + y + 3, 1)) is False
    assert (rootof(x ** 3 + x + 3, 1) == rootof(y ** 3 + y + 3, 1)) is True
    assert (rootof(x ** 3 + x + 3, 1) == rootof(y ** 3 + y + 3, 2)) is False
    assert (rootof(x ** 3 + x + 3, 2) == rootof(y ** 3 + y + 3, 2)) is True