from sympy.polys.densebasic import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring
from sympy.core.singleton import S
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_dmp_zero_p():
    assert dmp_zero_p([], 0) is True
    assert dmp_zero_p([[]], 1) is True
    assert dmp_zero_p([[[]]], 2) is True
    assert dmp_zero_p([[[1]]], 2) is False