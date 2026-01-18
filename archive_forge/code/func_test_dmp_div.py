from sympy.polys.densebasic import (
from sympy.polys.densearith import (
from sympy.polys.polyerrors import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import FF, ZZ, QQ
from sympy.testing.pytest import raises
def test_dmp_div():
    f, g, q, r = ([5, 4, 3, 2, 1], [1, 2, 3], [5, -6, 0], [20, 1])
    assert dmp_div(f, g, 0, ZZ) == (q, r)
    assert dmp_quo(f, g, 0, ZZ) == q
    assert dmp_rem(f, g, 0, ZZ) == r
    raises(ExactQuotientFailed, lambda: dmp_exquo(f, g, 0, ZZ))
    f, g, q, r = ([[[1]]], [[[2]], [1]], [[[]]], [[[1]]])
    assert dmp_div(f, g, 2, ZZ) == (q, r)
    assert dmp_quo(f, g, 2, ZZ) == q
    assert dmp_rem(f, g, 2, ZZ) == r
    raises(ExactQuotientFailed, lambda: dmp_exquo(f, g, 2, ZZ))