from sympy.polys.densebasic import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring
from sympy.core.singleton import S
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_dmp_grounds():
    assert dmp_grounds(ZZ(7), 0, 2) == []
    assert dmp_grounds(ZZ(7), 1, 2) == [[[[7]]]]
    assert dmp_grounds(ZZ(7), 2, 2) == [[[[7]]], [[[7]]]]
    assert dmp_grounds(ZZ(7), 3, 2) == [[[[7]]], [[[7]]], [[[7]]]]
    assert dmp_grounds(ZZ(7), 3, -1) == [7, 7, 7]