from sympy.polys.densebasic import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring
from sympy.core.singleton import S
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_dmp_ground_p():
    assert dmp_ground_p([], 0, 0) is True
    assert dmp_ground_p([[]], 0, 1) is True
    assert dmp_ground_p([[]], 1, 1) is False
    assert dmp_ground_p([[ZZ(1)]], 1, 1) is True
    assert dmp_ground_p([[[ZZ(2)]]], 2, 2) is True
    assert dmp_ground_p([[[ZZ(2)]]], 3, 2) is False
    assert dmp_ground_p([[[ZZ(3)], []]], 3, 2) is False
    assert dmp_ground_p([], None, 0) is True
    assert dmp_ground_p([[]], None, 1) is True
    assert dmp_ground_p([ZZ(1)], None, 0) is True
    assert dmp_ground_p([[[ZZ(1)]]], None, 2) is True
    assert dmp_ground_p([[[ZZ(3)], []]], None, 2) is False