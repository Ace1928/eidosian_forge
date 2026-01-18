from sympy.polys.densebasic import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring
from sympy.core.singleton import S
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_dmp_ground_nth():
    assert dmp_ground_nth([[]], (0, 0), 1, ZZ) == 0
    assert dmp_ground_nth([[1], [2], [3]], (0, 0), 1, ZZ) == 3
    assert dmp_ground_nth([[1], [2], [3]], (1, 0), 1, ZZ) == 2
    assert dmp_ground_nth([[1], [2], [3]], (2, 0), 1, ZZ) == 1
    assert dmp_ground_nth([[1], [2], [3]], (2, 1), 1, ZZ) == 0
    assert dmp_ground_nth([[1], [2], [3]], (3, 0), 1, ZZ) == 0
    raises(IndexError, lambda: dmp_ground_nth([[3], [4], [5]], (2, -1), 1, ZZ))