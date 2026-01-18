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
def test_diopcoverage():
    eq = (2 * x + y + 1) ** 2
    assert diop_solve(eq) == {(t_0, -2 * t_0 - 1)}
    eq = 2 * x ** 2 + 6 * x * y + 12 * x + 4 * y ** 2 + 18 * y + 18
    assert diop_solve(eq) == {(t, -t - 3), (2 * t - 3, -t)}
    assert diop_quadratic(x + y ** 2 - 3) == {(-t ** 2 + 3, -t)}
    assert diop_linear(x + y - 3) == (t_0, 3 - t_0)
    assert base_solution_linear(0, 1, 2, t=None) == (0, 0)
    ans = (3 * t - 1, -2 * t + 1)
    assert base_solution_linear(4, 8, 12, t) == ans
    assert base_solution_linear(4, 8, 12, t=None) == tuple((_.subs(t, 0) for _ in ans))
    assert cornacchia(1, 1, 20) is None
    assert cornacchia(1, 1, 5) == {(2, 1)}
    assert cornacchia(1, 2, 17) == {(3, 2)}
    raises(ValueError, lambda: reconstruct(4, 20, 1))
    assert gaussian_reduce(4, 1, 3) == (1, 1)
    eq = -w ** 2 - x ** 2 - y ** 2 + z ** 2
    assert diop_general_pythagorean(eq) == diop_general_pythagorean(-eq) == (m1 ** 2 + m2 ** 2 - m3 ** 2, 2 * m1 * m3, 2 * m2 * m3, m1 ** 2 + m2 ** 2 + m3 ** 2)
    assert len(check_param(S(3) + x / 3, S(4) + x / 2, S(2), [x])) == 0
    assert len(check_param(Rational(3, 2), S(4) + x, S(2), [x])) == 0
    assert len(check_param(S(4) + x, Rational(3, 2), S(2), [x])) == 0
    assert _nint_or_floor(16, 10) == 2
    assert _odd(1) == (not _even(1)) == True
    assert _odd(0) == (not _even(0)) == False
    assert _remove_gcd(2, 4, 6) == (1, 2, 3)
    raises(TypeError, lambda: _remove_gcd((2, 4, 6)))
    assert sqf_normal(2 * 3 ** 2 * 5, 2 * 5 * 11, 2 * 7 ** 2 * 11) == (11, 1, 5)
    raises(NotImplementedError, lambda: diophantine(x ** 2 + y ** 2 + x * y + 2 * y * z - 12))
    raises(NotImplementedError, lambda: diophantine(x ** 3 + y ** 2))
    assert diop_quadratic(x ** 2 + y ** 2 - 1 ** 2 - 3 ** 4) == {(-9, -1), (-9, 1), (-1, -9), (-1, 9), (1, -9), (1, 9), (9, -1), (9, 1)}