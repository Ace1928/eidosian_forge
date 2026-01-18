from sympy.polys.densebasic import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring
from sympy.core.singleton import S
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_dmp_permute():
    f = dmp_normal([[1, 0, 0], [], [1, 0], [], [1]], 1, ZZ)
    g = dmp_normal([[1, 0, 0, 0, 0], [1, 0, 0], [1]], 1, ZZ)
    assert dmp_permute(f, [0, 1], 1, ZZ) == f
    assert dmp_permute(g, [0, 1], 1, ZZ) == g
    assert dmp_permute(f, [1, 0], 1, ZZ) == g
    assert dmp_permute(g, [1, 0], 1, ZZ) == f