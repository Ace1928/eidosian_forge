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
def test_diop_partition():
    for n in [8, 10]:
        for k in range(1, 8):
            for p in partition(n, k):
                assert len(p) == k
    assert list(partition(3, 5)) == []
    assert [list(p) for p in partition(3, 5, 1)] == [[0, 0, 0, 0, 3], [0, 0, 0, 1, 2], [0, 0, 1, 1, 1]]
    assert list(partition(0)) == [()]
    assert list(partition(1, 0)) == [()]
    assert [list(i) for i in partition(3)] == [[1, 1, 1], [1, 2], [3]]