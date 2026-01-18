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
def test_diophantine_solution_set():
    s1 = DiophantineSolutionSet([], [])
    assert set(s1) == set()
    assert s1.symbols == ()
    assert s1.parameters == ()
    raises(ValueError, lambda: s1.add((x,)))
    assert list(s1.dict_iterator()) == []
    s2 = DiophantineSolutionSet([x, y], [t, u])
    assert s2.symbols == (x, y)
    assert s2.parameters == (t, u)
    raises(ValueError, lambda: s2.add((1,)))
    s2.add((3, 4))
    assert set(s2) == {(3, 4)}
    s2.update((3, 4), (-1, u))
    assert set(s2) == {(3, 4), (-1, u)}
    raises(ValueError, lambda: s1.update(s2))
    assert list(s2.dict_iterator()) == [{x: -1, y: u}, {x: 3, y: 4}]
    s3 = DiophantineSolutionSet([x, y, z], [t, u])
    assert len(s3.parameters) == 2
    s3.add((t ** 2 + u, t - u, 1))
    assert set(s3) == {(t ** 2 + u, t - u, 1)}
    assert s3.subs(t, 2) == {(u + 4, 2 - u, 1)}
    assert s3(2) == {(u + 4, 2 - u, 1)}
    assert s3.subs({t: 7, u: 8}) == {(57, -1, 1)}
    assert s3(7, 8) == {(57, -1, 1)}
    assert s3.subs({t: 5}) == {(u + 25, 5 - u, 1)}
    assert s3(5) == {(u + 25, 5 - u, 1)}
    assert s3.subs(u, -3) == {(t ** 2 - 3, t + 3, 1)}
    assert s3(None, -3) == {(t ** 2 - 3, t + 3, 1)}
    assert s3.subs({t: 2, u: 8}) == {(12, -6, 1)}
    assert s3(2, 8) == {(12, -6, 1)}
    assert s3.subs({t: 5, u: -3}) == {(22, 8, 1)}
    assert s3(5, -3) == {(22, 8, 1)}
    raises(ValueError, lambda: s3.subs(x=1))
    raises(ValueError, lambda: s3.subs(1, 2, 3))
    raises(ValueError, lambda: s3.add(()))
    raises(ValueError, lambda: s3.add((1, 2, 3, 4)))
    raises(ValueError, lambda: s3.add((1, 2)))
    raises(ValueError, lambda: s3(1, 2, 3))
    raises(TypeError, lambda: s3(t=1))
    s4 = DiophantineSolutionSet([x, y], [t, u])
    s4.add((t, 11 * t))
    s4.add((-t, 22 * t))
    assert s4(0, 0) == {(0, 0)}