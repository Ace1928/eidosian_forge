from sympy.polys.densebasic import (
from sympy.polys.densearith import (
from sympy.polys.polyerrors import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import FF, ZZ, QQ
from sympy.testing.pytest import raises
def test_dup_expand():
    assert dup_expand((), ZZ) == [1]
    assert dup_expand(([1, 2, 3], [1, 2], [7, 5, 4, 3]), ZZ) == dup_mul([1, 2, 3], dup_mul([1, 2], [7, 5, 4, 3], ZZ), ZZ)