from sympy.polys.densebasic import (
from sympy.polys.densearith import (
from sympy.polys.polyerrors import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import FF, ZZ, QQ
from sympy.testing.pytest import raises
def test_dmp_add_term():
    assert dmp_add_term([ZZ(1), ZZ(1), ZZ(1)], ZZ(1), 2, 0, ZZ) == dup_add_term([ZZ(1), ZZ(1), ZZ(1)], ZZ(1), 2, ZZ)
    assert dmp_add_term(f_0, [[]], 3, 2, ZZ) == f_0
    assert dmp_add_term(F_0, [[]], 3, 2, QQ) == F_0