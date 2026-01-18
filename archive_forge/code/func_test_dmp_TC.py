from sympy.polys.densebasic import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring
from sympy.core.singleton import S
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_dmp_TC():
    assert dmp_TC([[]], ZZ) == []
    assert dmp_TC([[2, 3, 4], [5]], ZZ) == [5]
    assert dmp_TC([[[]]], ZZ) == [[]]
    assert dmp_TC([[[2], [3, 4]], [[5]]], ZZ) == [[5]]