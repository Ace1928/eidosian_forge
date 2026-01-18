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
def test_dmp_trunc():
    assert dmp_trunc([[]], [1, 2], 2, ZZ) == [[]]
    assert dmp_trunc([[1, 2], [1, 4, 1], [1]], [1, 2], 1, ZZ) == [[-3], [1]]