from sympy.polys.densebasic import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring
from sympy.core.singleton import S
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_dmp_degree():
    assert dmp_degree([[]], 1) is -oo
    assert dmp_degree([[[]]], 2) is -oo
    assert dmp_degree([[1]], 1) == 0
    assert dmp_degree([[2], [1]], 1) == 1