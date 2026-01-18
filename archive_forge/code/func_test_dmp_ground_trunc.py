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
def test_dmp_ground_trunc():
    assert dmp_ground_trunc(f_0, ZZ(3), 2, ZZ) == dmp_normal([[[1, -1, 0], [-1]], [[]], [[1, -1, 0], [1, -1, 1], [1]]], 2, ZZ)