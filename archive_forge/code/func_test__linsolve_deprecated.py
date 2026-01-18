from sympy.testing.pytest import raises
from sympy.core.numbers import I
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.abc import x, y, z
from sympy.polys.matrices.linsolve import _linsolve
from sympy.polys.solvers import PolyNonlinearError
def test__linsolve_deprecated():
    raises(PolyNonlinearError, lambda: _linsolve([Eq(x ** 2, x ** 2 + y)], [x, y]))
    raises(PolyNonlinearError, lambda: _linsolve([(x + y) ** 2 - x ** 2], [x]))
    raises(PolyNonlinearError, lambda: _linsolve([Eq((x + y) ** 2, x ** 2)], [x]))