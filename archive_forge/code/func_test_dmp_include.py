from sympy.polys.densebasic import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring
from sympy.core.singleton import S
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_dmp_include():
    assert dmp_include([1, 2, 3], [], 0, ZZ) == [1, 2, 3]
    assert dmp_include([1, 2, 3], [0], 0, ZZ) == [[1, 2, 3]]
    assert dmp_include([1, 2, 3], [1], 0, ZZ) == [[1], [2], [3]]
    assert dmp_include([1, 2, 3], [0, 1], 0, ZZ) == [[[1, 2, 3]]]
    assert dmp_include([1, 2, 3], [1, 2], 0, ZZ) == [[[1]], [[2]], [[3]]]