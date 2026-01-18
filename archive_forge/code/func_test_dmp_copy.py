from sympy.polys.densebasic import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring
from sympy.core.singleton import S
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_dmp_copy():
    f = [[ZZ(1)], [ZZ(2), ZZ(0)]]
    g = dmp_copy(f, 1)
    g[0][0], g[1][1] = (ZZ(7), ZZ(1))
    assert f != g