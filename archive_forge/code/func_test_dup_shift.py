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
def test_dup_shift():
    assert dup_shift([], 1, ZZ) == []
    assert dup_shift([1], 1, ZZ) == [1]
    assert dup_shift([1, 2, 3, 4, 5], 1, ZZ) == [1, 6, 15, 20, 15]
    assert dup_shift([1, 2, 3, 4, 5], 7, ZZ) == [1, 30, 339, 1712, 3267]