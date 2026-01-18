from sympy.polys.densebasic import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring
from sympy.core.singleton import S
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_dmp_raise():
    assert dmp_raise([], 2, 0, ZZ) == [[[]]]
    assert dmp_raise([[1]], 0, 1, ZZ) == [[1]]
    assert dmp_raise([[1, 2, 3], [], [2, 3]], 2, 1, ZZ) == [[[[1]], [[2]], [[3]]], [[[]]], [[[2]], [[3]]]]