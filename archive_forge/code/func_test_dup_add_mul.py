from sympy.polys.densebasic import (
from sympy.polys.densearith import (
from sympy.polys.polyerrors import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import FF, ZZ, QQ
from sympy.testing.pytest import raises
def test_dup_add_mul():
    assert dup_add_mul([ZZ(1), ZZ(2), ZZ(3)], [ZZ(3), ZZ(2), ZZ(1)], [ZZ(1), ZZ(2)], ZZ) == [ZZ(3), ZZ(9), ZZ(7), ZZ(5)]
    assert dmp_add_mul([[ZZ(1), ZZ(2)], [ZZ(3)]], [[ZZ(3)], [ZZ(2), ZZ(1)]], [[ZZ(1)], [ZZ(2)]], 1, ZZ) == [[ZZ(3)], [ZZ(3), ZZ(9)], [ZZ(4), ZZ(5)]]