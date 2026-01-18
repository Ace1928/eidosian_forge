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
def test_dup_sign_variations():
    assert dup_sign_variations([], ZZ) == 0
    assert dup_sign_variations([1, 0], ZZ) == 0
    assert dup_sign_variations([1, 0, 2], ZZ) == 0
    assert dup_sign_variations([1, 0, 3, 0], ZZ) == 0
    assert dup_sign_variations([1, 0, 4, 0, 5], ZZ) == 0
    assert dup_sign_variations([-1, 0, 2], ZZ) == 1
    assert dup_sign_variations([-1, 0, 3, 0], ZZ) == 1
    assert dup_sign_variations([-1, 0, 4, 0, 5], ZZ) == 1
    assert dup_sign_variations([-1, -4, -5], ZZ) == 0
    assert dup_sign_variations([1, -4, -5], ZZ) == 1
    assert dup_sign_variations([1, 4, -5], ZZ) == 1
    assert dup_sign_variations([1, -4, 5], ZZ) == 2
    assert dup_sign_variations([-1, 4, -5], ZZ) == 2
    assert dup_sign_variations([-1, 4, 5], ZZ) == 1
    assert dup_sign_variations([-1, -4, 5], ZZ) == 1
    assert dup_sign_variations([1, 4, 5], ZZ) == 0
    assert dup_sign_variations([-1, 0, -4, 0, -5], ZZ) == 0
    assert dup_sign_variations([1, 0, -4, 0, -5], ZZ) == 1
    assert dup_sign_variations([1, 0, 4, 0, -5], ZZ) == 1
    assert dup_sign_variations([1, 0, -4, 0, 5], ZZ) == 2
    assert dup_sign_variations([-1, 0, 4, 0, -5], ZZ) == 2
    assert dup_sign_variations([-1, 0, 4, 0, 5], ZZ) == 1
    assert dup_sign_variations([-1, 0, -4, 0, 5], ZZ) == 1
    assert dup_sign_variations([1, 0, 4, 0, 5], ZZ) == 0