from sympy.polys.densebasic import (
from sympy.polys.densearith import (
from sympy.polys.polyerrors import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import FF, ZZ, QQ
from sympy.testing.pytest import raises
def test_dup_max_norm():
    assert dup_max_norm([], ZZ) == 0
    assert dup_max_norm([1], ZZ) == 1
    assert dup_max_norm([1, 4, 2, 3], ZZ) == 4