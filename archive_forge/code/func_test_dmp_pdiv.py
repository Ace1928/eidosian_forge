from sympy.polys.densebasic import (
from sympy.polys.densearith import (
from sympy.polys.polyerrors import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import FF, ZZ, QQ
from sympy.testing.pytest import raises
def test_dmp_pdiv():
    f = dmp_normal([[1], [], [1, 0, 0]], 1, ZZ)
    g = dmp_normal([[1], [-1, 0]], 1, ZZ)
    q = dmp_normal([[1], [1, 0]], 1, ZZ)
    r = dmp_normal([[2, 0, 0]], 1, ZZ)
    assert dmp_pdiv(f, g, 1, ZZ) == (q, r)
    assert dmp_pquo(f, g, 1, ZZ) == q
    assert dmp_prem(f, g, 1, ZZ) == r
    raises(ExactQuotientFailed, lambda: dmp_pexquo(f, g, 1, ZZ))
    f = dmp_normal([[1], [], [1, 0, 0]], 1, ZZ)
    g = dmp_normal([[2], [-2, 0]], 1, ZZ)
    q = dmp_normal([[2], [2, 0]], 1, ZZ)
    r = dmp_normal([[8, 0, 0]], 1, ZZ)
    assert dmp_pdiv(f, g, 1, ZZ) == (q, r)
    assert dmp_pquo(f, g, 1, ZZ) == q
    assert dmp_prem(f, g, 1, ZZ) == r
    raises(ExactQuotientFailed, lambda: dmp_pexquo(f, g, 1, ZZ))