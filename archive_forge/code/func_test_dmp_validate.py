from sympy.polys.densebasic import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring
from sympy.core.singleton import S
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_dmp_validate():
    assert dmp_validate([]) == ([], 0)
    assert dmp_validate([0, 0, 0, 1, 0]) == ([1, 0], 0)
    assert dmp_validate([[[]]]) == ([[[]]], 2)
    assert dmp_validate([[0], [], [0], [1], [0]]) == ([[1], []], 1)
    raises(ValueError, lambda: dmp_validate([[0], 0, [0], [1], [0]]))