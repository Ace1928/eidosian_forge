from sympy.polys.densebasic import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring
from sympy.core.singleton import S
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_dup_inflate():
    assert dup_inflate([], 17, ZZ) == []
    assert dup_inflate([1, 2, 3], 1, ZZ) == [1, 2, 3]
    assert dup_inflate([1, 2, 3], 2, ZZ) == [1, 0, 2, 0, 3]
    assert dup_inflate([1, 2, 3], 3, ZZ) == [1, 0, 0, 2, 0, 0, 3]
    assert dup_inflate([1, 2, 3], 4, ZZ) == [1, 0, 0, 0, 2, 0, 0, 0, 3]
    raises(IndexError, lambda: dup_inflate([1, 2, 3], 0, ZZ))