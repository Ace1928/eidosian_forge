from sympy.polys.densebasic import (
from sympy.polys.densearith import (
from sympy.polys.polyerrors import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import FF, ZZ, QQ
from sympy.testing.pytest import raises
def test_dmp_expand():
    assert dmp_expand((), 1, ZZ) == [[1]]
    assert dmp_expand(([[1], [2], [3]], [[1], [2]], [[7], [5], [4], [3]]), 1, ZZ) == dmp_mul([[1], [2], [3]], dmp_mul([[1], [2]], [[7], [5], [4], [3]], 1, ZZ), 1, ZZ)