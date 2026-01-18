from sympy.testing.pytest import raises
from sympy.core.numbers import I
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.abc import x, y, z
from sympy.polys.matrices.linsolve import _linsolve
from sympy.polys.solvers import PolyNonlinearError
def all_close(sol1, sol2, eps=1e-15):
    close = lambda a, b: abs(a - b) < eps
    assert sol1.keys() == sol2.keys()
    return all((close(sol1[s], sol2[s]) for s in sol1))