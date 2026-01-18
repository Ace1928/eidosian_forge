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
def test_dup_decompose():
    assert dup_decompose([1], ZZ) == [[1]]
    assert dup_decompose([1, 0], ZZ) == [[1, 0]]
    assert dup_decompose([1, 0, 0, 0], ZZ) == [[1, 0, 0, 0]]
    assert dup_decompose([1, 0, 0, 0, 0], ZZ) == [[1, 0, 0], [1, 0, 0]]
    assert dup_decompose([1, 0, 0, 0, 0, 0, 0], ZZ) == [[1, 0, 0, 0], [1, 0, 0]]
    assert dup_decompose([7, 0, 0, 0, 1], ZZ) == [[7, 0, 1], [1, 0, 0]]
    assert dup_decompose([4, 0, 3, 0, 2], ZZ) == [[4, 3, 2], [1, 0, 0]]
    f = [1, 0, 20, 0, 150, 0, 500, 0, 625, -2, 0, -10, 9]
    assert dup_decompose(f, ZZ) == [[1, 0, 0, -2, 9], [1, 0, 5, 0]]
    f = [2, 0, 40, 0, 300, 0, 1000, 0, 1250, -4, 0, -20, 18]
    assert dup_decompose(f, ZZ) == [[2, 0, 0, -4, 18], [1, 0, 5, 0]]
    f = [1, 0, 20, -8, 150, -120, 524, -600, 865, -1034, 600, -170, 29]
    assert dup_decompose(f, ZZ) == [[1, -8, 24, -34, 29], [1, 0, 5, 0]]
    R, t = ring('t', ZZ)
    f = [6 * t ** 2 - 42, 48 * t ** 2 + 96, 144 * t ** 2 + 648 * t + 288, 624 * t ** 2 + 864 * t + 384, 108 * t ** 3 + 312 * t ** 2 + 432 * t + 192]
    assert dup_decompose(f, R.to_domain()) == [f]