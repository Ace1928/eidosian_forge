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
def test_dup_real_imag():
    assert dup_real_imag([], ZZ) == ([[]], [[]])
    assert dup_real_imag([1], ZZ) == ([[1]], [[]])
    assert dup_real_imag([1, 1], ZZ) == ([[1], [1]], [[1, 0]])
    assert dup_real_imag([1, 2], ZZ) == ([[1], [2]], [[1, 0]])
    assert dup_real_imag([1, 2, 3], ZZ) == ([[1], [2], [-1, 0, 3]], [[2, 0], [2, 0]])
    raises(DomainError, lambda: dup_real_imag([EX(1), EX(2)], EX))