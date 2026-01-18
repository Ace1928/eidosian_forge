from sympy.polys.densebasic import (
from sympy.polys.densearith import (
from sympy.polys.polyerrors import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import FF, ZZ, QQ
from sympy.testing.pytest import raises
def test_dmp_quo_ground():
    f = dmp_normal([[6], [2], [8]], 1, ZZ)
    assert dmp_quo_ground(f, ZZ(1), 1, ZZ) == f
    assert dmp_quo_ground(f, ZZ(2), 1, ZZ) == dmp_normal([[3], [1], [4]], 1, ZZ)
    assert dmp_normal(dmp_quo_ground(f, ZZ(3), 1, ZZ), 1, ZZ) == dmp_normal([[2], [], [2]], 1, ZZ)