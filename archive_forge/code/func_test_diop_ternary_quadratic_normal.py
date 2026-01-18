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
def test_diop_ternary_quadratic_normal():
    assert check_solutions(234 * x ** 2 - 65601 * y ** 2 - z ** 2)
    assert check_solutions(23 * x ** 2 + 616 * y ** 2 - z ** 2)
    assert check_solutions(5 * x ** 2 + 4 * y ** 2 - z ** 2)
    assert check_solutions(3 * x ** 2 + 6 * y ** 2 - 3 * z ** 2)
    assert check_solutions(x ** 2 + 3 * y ** 2 - z ** 2)
    assert check_solutions(4 * x ** 2 + 5 * y ** 2 - z ** 2)
    assert check_solutions(x ** 2 + y ** 2 - z ** 2)
    assert check_solutions(16 * x ** 2 + y ** 2 - 25 * z ** 2)
    assert check_solutions(6 * x ** 2 - y ** 2 + 10 * z ** 2)
    assert check_solutions(213 * x ** 2 + 12 * y ** 2 - 9 * z ** 2)
    assert check_solutions(34 * x ** 2 - 3 * y ** 2 - 301 * z ** 2)
    assert check_solutions(124 * x ** 2 - 30 * y ** 2 - 7729 * z ** 2)