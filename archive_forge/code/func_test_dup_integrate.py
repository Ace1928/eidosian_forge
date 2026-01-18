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
def test_dup_integrate():
    assert dup_integrate([], 1, QQ) == []
    assert dup_integrate([], 2, QQ) == []
    assert dup_integrate([QQ(1)], 1, QQ) == [QQ(1), QQ(0)]
    assert dup_integrate([QQ(1)], 2, QQ) == [QQ(1, 2), QQ(0), QQ(0)]
    assert dup_integrate([QQ(1), QQ(2), QQ(3)], 0, QQ) == [QQ(1), QQ(2), QQ(3)]
    assert dup_integrate([QQ(1), QQ(2), QQ(3)], 1, QQ) == [QQ(1, 3), QQ(1), QQ(3), QQ(0)]
    assert dup_integrate([QQ(1), QQ(2), QQ(3)], 2, QQ) == [QQ(1, 12), QQ(1, 3), QQ(3, 2), QQ(0), QQ(0)]
    assert dup_integrate([QQ(1), QQ(2), QQ(3)], 3, QQ) == [QQ(1, 60), QQ(1, 12), QQ(1, 2), QQ(0), QQ(0), QQ(0)]
    assert dup_integrate(dup_from_raw_dict({29: QQ(17)}, QQ), 3, QQ) == dup_from_raw_dict({32: QQ(17, 29760)}, QQ)
    assert dup_integrate(dup_from_raw_dict({29: QQ(17), 5: QQ(1, 2)}, QQ), 3, QQ) == dup_from_raw_dict({32: QQ(17, 29760), 8: QQ(1, 672)}, QQ)