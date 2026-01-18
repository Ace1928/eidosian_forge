from sympy.polys.densebasic import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring
from sympy.core.singleton import S
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_dup_copy():
    f = [ZZ(1), ZZ(0), ZZ(2)]
    g = dup_copy(f)
    g[0], g[2] = (ZZ(7), ZZ(0))
    assert f != g