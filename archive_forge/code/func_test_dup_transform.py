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
def test_dup_transform():
    assert dup_transform([], [], [1, 1], ZZ) == []
    assert dup_transform([], [1], [1, 1], ZZ) == []
    assert dup_transform([], [1, 2], [1, 1], ZZ) == []
    assert dup_transform([6, -5, 4, -3, 17], [1, -3, 4], [2, -3], ZZ) == [6, -82, 541, -2205, 6277, -12723, 17191, -13603, 4773]