from sympy.polys.densebasic import (
from sympy.polys.densearith import (
from sympy.polys.polyerrors import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import FF, ZZ, QQ
from sympy.testing.pytest import raises
def test_dup_mul_ground():
    f = dup_normal([], ZZ)
    assert dup_mul_ground(f, ZZ(2), ZZ) == dup_normal([], ZZ)
    f = dup_normal([1, 2, 3], ZZ)
    assert dup_mul_ground(f, ZZ(0), ZZ) == dup_normal([], ZZ)
    assert dup_mul_ground(f, ZZ(2), ZZ) == dup_normal([2, 4, 6], ZZ)