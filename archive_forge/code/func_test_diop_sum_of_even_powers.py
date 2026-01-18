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
def test_diop_sum_of_even_powers():
    eq = x ** 4 + y ** 4 + z ** 4 - 2673
    assert diop_solve(eq) == {(3, 6, 6), (2, 4, 7)}
    assert diop_general_sum_of_even_powers(eq, 2) == {(3, 6, 6), (2, 4, 7)}
    raises(NotImplementedError, lambda: diop_general_sum_of_even_powers(-eq, 2))
    neg = symbols('neg', negative=True)
    eq = x ** 4 + y ** 4 + neg ** 4 - 2673
    assert diop_general_sum_of_even_powers(eq) == {(-3, 6, 6)}
    assert diophantine(x ** 4 + y ** 4 + 2) == set()
    assert diop_general_sum_of_even_powers(x ** 4 + y ** 4 - 2, limit=0) == set()