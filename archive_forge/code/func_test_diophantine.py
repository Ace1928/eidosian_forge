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
def test_diophantine():
    assert check_solutions((x - y) * (y - z) * (z - x))
    assert check_solutions((x - y) * (x ** 2 + y ** 2 - z ** 2))
    assert check_solutions((x - 3 * y + 7 * z) * (x ** 2 + y ** 2 - z ** 2))
    assert check_solutions(x ** 2 - 3 * y ** 2 - 1)
    assert check_solutions(y ** 2 + 7 * x * y)
    assert check_solutions(x ** 2 - 3 * x * y + y ** 2)
    assert check_solutions(z * (x ** 2 - y ** 2 - 15))
    assert check_solutions(x * (2 * y - 2 * z + 5))
    assert check_solutions((x ** 2 - 3 * y ** 2 - 1) * (x ** 2 - y ** 2 - 15))
    assert check_solutions((x ** 2 - 3 * y ** 2 - 1) * (y - 7 * z))
    assert check_solutions((x ** 2 + y ** 2 - z ** 2) * (x - 7 * y - 3 * z + 4 * w))
    assert check_solutions(y ** 2 - 7 * x * y + 4 * y * z)
    assert check_solutions(x ** 2 - 2 * x + 1)
    assert diophantine(x - y) == diophantine(Eq(x, y))
    eq = x ** 4 + y ** 4 - 97
    assert diophantine(eq, permute=True) == diophantine(-eq, permute=True)
    assert diophantine(3 * x * pi - 2 * y * pi) == {(2 * t_0, 3 * t_0)}
    eq = x ** 2 + y ** 2 + z ** 2 - 14
    base_sol = {(1, 2, 3)}
    assert diophantine(eq) == base_sol
    complete_soln = set(signed_permutations(base_sol.pop()))
    assert diophantine(eq, permute=True) == complete_soln
    assert diophantine(x ** 2 + x * Rational(15, 14) - 3) == set()
    eq = 92 * x ** 2 - 99 * y ** 2 - z ** 2
    coeff = eq.as_coefficients_dict()
    assert _diop_ternary_quadratic_normal((x, y, z), coeff) == {(9, 7, 51)}
    assert diophantine(eq) == {(891 * p ** 2 + 9 * q ** 2, -693 * p ** 2 - 102 * p * q + 7 * q ** 2, 5049 * p ** 2 - 1386 * p * q - 51 * q ** 2)}
    eq = 2 * x ** 2 + 2 * y ** 2 - z ** 2
    coeff = eq.as_coefficients_dict()
    assert _diop_ternary_quadratic_normal((x, y, z), coeff) == {(1, 1, 2)}
    assert diophantine(eq) == {(2 * p ** 2 - q ** 2, -2 * p ** 2 + 4 * p * q - q ** 2, 4 * p ** 2 - 4 * p * q + 2 * q ** 2)}
    eq = 411 * x ** 2 + 57 * y ** 2 - 221 * z ** 2
    coeff = eq.as_coefficients_dict()
    assert _diop_ternary_quadratic_normal((x, y, z), coeff) == {(2021, 2645, 3066)}
    assert diophantine(eq) == {(115197 * p ** 2 - 446641 * q ** 2, -150765 * p ** 2 + 1355172 * p * q - 584545 * q ** 2, 174762 * p ** 2 - 301530 * p * q + 677586 * q ** 2)}
    eq = 573 * x ** 2 + 267 * y ** 2 - 984 * z ** 2
    coeff = eq.as_coefficients_dict()
    assert _diop_ternary_quadratic_normal((x, y, z), coeff) == {(49, 233, 127)}
    assert diophantine(eq) == {(4361 * p ** 2 - 16072 * q ** 2, -20737 * p ** 2 + 83312 * p * q - 76424 * q ** 2, 11303 * p ** 2 - 41474 * p * q + 41656 * q ** 2)}
    eq = x ** 2 + 3 * y ** 2 - 12 * z ** 2
    coeff = eq.as_coefficients_dict()
    assert _diop_ternary_quadratic_normal((x, y, z), coeff) == {(0, 2, 1)}
    assert diophantine(eq) == {(24 * p * q, 2 * p ** 2 - 24 * q ** 2, p ** 2 + 12 * q ** 2)}
    raises(NotImplementedError, lambda: diophantine(x * y ** 2 + 1))
    assert diophantine(1 / x) == set()
    assert diophantine(1 / x + 1 / y - S.Half) == {(6, 3), (-2, 1), (4, 4), (1, -2), (3, 6)}
    assert diophantine(x ** 2 + y ** 2 + 3 * x - 5, permute=True) == {(-1, 1), (-4, -1), (1, -1), (1, 1), (-4, 1), (-1, -1), (4, 1), (4, -1)}
    assert diophantine(y ** 4 + x ** 4 - 2 ** 4 - 3 ** 4, syms=(x, y), permute=True) == {(-3, -2), (-3, 2), (-2, -3), (-2, 3), (2, -3), (2, 3), (3, -2), (3, 2)}
    assert diophantine(y ** 4 + x ** 4 - 2 ** 4 - 3 ** 4, syms=(y, x), permute=True) == {(-3, -2), (-3, 2), (-2, -3), (-2, 3), (2, -3), (2, 3), (3, -2), (3, 2)}
    assert check_solutions(x ** 2 - y)
    assert check_solutions(y ** 2 - x)
    assert diophantine(x ** 2 - y, t) == {(t, t ** 2)}
    assert diophantine(y ** 2 - x, t) == {(t ** 2, -t)}