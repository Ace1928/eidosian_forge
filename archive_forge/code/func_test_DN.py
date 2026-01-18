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
def test_DN():
    assert diop_DN(3, 0) == [(0, 0)]
    assert diop_DN(-17, -5) == []
    assert diop_DN(-19, 23) == [(2, 1)]
    assert diop_DN(-13, 17) == [(2, 1)]
    assert diop_DN(-15, 13) == []
    assert diop_DN(0, 5) == []
    assert diop_DN(0, 9) == [(3, t)]
    assert diop_DN(9, 0) == [(3 * t, t)]
    assert diop_DN(16, 24) == []
    assert diop_DN(9, 180) == [(18, 4)]
    assert diop_DN(9, -180) == [(12, 6)]
    assert diop_DN(7, 0) == [(0, 0)]
    assert diop_DN(-1, 5) == [(2, 1), (1, 2)]
    assert diop_DN(-1, 169) == [(12, 5), (5, 12), (13, 0), (0, 13)]
    assert diop_DN(13, 1) == [(649, 180)]
    assert diop_DN(980, 1) == [(51841, 1656)]
    assert diop_DN(981, 1) == [(158070671986249, 5046808151700)]
    assert diop_DN(986, 1) == [(49299, 1570)]
    assert diop_DN(991, 1) == [(379516400906811930638014896080, 12055735790331359447442538767)]
    assert diop_DN(17, 1) == [(33, 8)]
    assert diop_DN(19, 1) == [(170, 39)]
    assert diop_DN(13, -1) == [(18, 5)]
    assert diop_DN(991, -1) == []
    assert diop_DN(41, -1) == [(32, 5)]
    assert diop_DN(290, -1) == [(17, 1)]
    assert diop_DN(21257, -1) == [(13913102721304, 95427381109)]
    assert diop_DN(32, -1) == []
    assert diop_DN(13, -4) == [(3, 1), (393, 109), (36, 10)]
    assert equivalent(-3, 1, 393, 109, 13, -4) == True
    assert diop_DN(13, 27) == [(220, 61), (40, 11), (768, 213), (12, 3)]
    assert set(diop_DN(157, 12)) == {(13, 1), (10663, 851), (579160, 46222), (483790960, 38610722), (26277068347, 2097138361), (21950079635497, 1751807067011)}
    assert diop_DN(13, 25) == [(3245, 900)]
    assert diop_DN(192, 18) == []
    assert diop_DN(23, 13) == [(-6, 1), (6, 1)]
    assert diop_DN(167, 2) == [(13, 1)]
    assert diop_DN(167, -2) == []
    assert diop_DN(123, -2) == [(11, 1)]
    assert equivalent(11, 1, -11, 1, 123, -2)
    assert diop_DN(123, -23) == [(-10, 1), (10, 1)]
    assert diop_DN(0, 0, t) == [(0, t)]
    assert diop_DN(0, -1, t) == []