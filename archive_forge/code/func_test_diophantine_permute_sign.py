from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.matrices.dense import Matrix
from sympy.ntheory.factor_ import factorint
from sympy.simplify.powsimp import powsimp
from sympy.core.function import _mexpand
from sympy.core.sorting import default_sort_key, ordered
from sympy.functions.elementary.trigonometric import sin
from sympy.solvers.diophantine import diophantine
from sympy.solvers.diophantine.diophantine import (diop_DN,
from sympy.testing.pytest import slow, raises, XFAIL
from sympy.utilities.iterables import (
def test_diophantine_permute_sign():
    from sympy.abc import a, b, c, d, e
    eq = a ** 4 + b ** 4 - (2 ** 4 + 3 ** 4)
    base_sol = {(2, 3)}
    assert diophantine(eq) == base_sol
    complete_soln = set(signed_permutations(base_sol.pop()))
    assert diophantine(eq, permute=True) == complete_soln
    eq = a ** 2 + b ** 2 + c ** 2 + d ** 2 + e ** 2 - 234
    assert len(diophantine(eq)) == 35
    assert len(diophantine(eq, permute=True)) == 62000
    soln = {(-1, -1), (-1, 2), (1, -2), (1, 1)}
    assert diophantine(10 * x ** 2 + 12 * x * y + 12 * y ** 2 - 34, permute=True) == soln