from sympy.polys.densebasic import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring
from sympy.core.singleton import S
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_dmp_nth():
    assert dmp_nth([[1], [2], [3]], 0, 1, ZZ) == [3]
    assert dmp_nth([[1], [2], [3]], 1, 1, ZZ) == [2]
    assert dmp_nth([[1], [2], [3]], 2, 1, ZZ) == [1]
    assert dmp_nth([[1], [2], [3]], 9, 1, ZZ) == []
    raises(IndexError, lambda: dmp_nth([[3], [4], [5]], -1, 1, ZZ))