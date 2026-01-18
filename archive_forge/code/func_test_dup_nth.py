from sympy.polys.densebasic import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring
from sympy.core.singleton import S
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_dup_nth():
    assert dup_nth([1, 2, 3], 0, ZZ) == 3
    assert dup_nth([1, 2, 3], 1, ZZ) == 2
    assert dup_nth([1, 2, 3], 2, ZZ) == 1
    assert dup_nth([1, 2, 3], 9, ZZ) == 0
    raises(IndexError, lambda: dup_nth([3, 4, 5], -1, ZZ))