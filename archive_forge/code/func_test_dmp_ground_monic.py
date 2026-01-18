from sympy.polys.densebasic import (
from sympy.polys.densearith import dmp_mul_ground
from sympy.polys.densetools import (
from sympy.polys.polyclasses import ANP
from sympy.polys.polyerrors import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import FF, ZZ, QQ, EX
from sympy.polys.rings import ring
from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.functions.elementary.trigonometric import sin
from sympy.abc import x
from sympy.testing.pytest import raises
def test_dmp_ground_monic():
    assert dmp_ground_monic([[3], [6], [9]], 1, ZZ) == [[1], [2], [3]]
    raises(ExactQuotientFailed, lambda: dmp_ground_monic([[3], [4], [5]], 1, ZZ))
    assert dmp_ground_monic([[]], 1, QQ) == [[]]
    assert dmp_ground_monic([[QQ(1)]], 1, QQ) == [[QQ(1)]]
    assert dmp_ground_monic([[QQ(7)], [QQ(1)], [QQ(21)]], 1, QQ) == [[QQ(1)], [QQ(1, 7)], [QQ(3)]]