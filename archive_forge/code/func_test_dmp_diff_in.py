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
def test_dmp_diff_in():
    assert dmp_diff_in(f_6, 2, 1, 3, ZZ) == dmp_swap(dmp_diff(dmp_swap(f_6, 0, 1, 3, ZZ), 2, 3, ZZ), 0, 1, 3, ZZ)
    assert dmp_diff_in(f_6, 3, 1, 3, ZZ) == dmp_swap(dmp_diff(dmp_swap(f_6, 0, 1, 3, ZZ), 3, 3, ZZ), 0, 1, 3, ZZ)
    assert dmp_diff_in(f_6, 2, 2, 3, ZZ) == dmp_swap(dmp_diff(dmp_swap(f_6, 0, 2, 3, ZZ), 2, 3, ZZ), 0, 2, 3, ZZ)
    assert dmp_diff_in(f_6, 3, 2, 3, ZZ) == dmp_swap(dmp_diff(dmp_swap(f_6, 0, 2, 3, ZZ), 3, 3, ZZ), 0, 2, 3, ZZ)