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
def test_dup_scale():
    assert dup_scale([], -1, ZZ) == []
    assert dup_scale([1], -1, ZZ) == [1]
    assert dup_scale([1, 2, 3, 4, 5], -1, ZZ) == [1, -2, 3, -4, 5]
    assert dup_scale([1, 2, 3, 4, 5], -7, ZZ) == [2401, -686, 147, -28, 5]