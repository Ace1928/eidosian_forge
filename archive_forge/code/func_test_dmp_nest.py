from sympy.polys.densebasic import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring
from sympy.core.singleton import S
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_dmp_nest():
    assert dmp_nest(ZZ(1), 2, ZZ) == [[[1]]]
    assert dmp_nest([[1]], 0, ZZ) == [[1]]
    assert dmp_nest([[1]], 1, ZZ) == [[[1]]]
    assert dmp_nest([[1]], 2, ZZ) == [[[[1]]]]