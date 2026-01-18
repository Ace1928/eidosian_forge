from sympy.polys.densebasic import (
from sympy.polys.densearith import (
from sympy.polys.polyerrors import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import FF, ZZ, QQ
from sympy.testing.pytest import raises
def test_dmp_ff_div():
    raises(ZeroDivisionError, lambda: dmp_ff_div([[1, 2], [3]], [[]], 1, QQ))
    f = dmp_normal([[1], [], [1, 0, 0]], 1, QQ)
    g = dmp_normal([[1], [-1, 0]], 1, QQ)
    q = [[QQ(1, 1)], [QQ(1, 1), QQ(0, 1)]]
    r = [[QQ(2, 1), QQ(0, 1), QQ(0, 1)]]
    assert dmp_ff_div(f, g, 1, QQ) == (q, r)
    f = dmp_normal([[1], [], [1, 0, 0]], 1, QQ)
    g = dmp_normal([[-1], [1, 0]], 1, QQ)
    q = [[QQ(-1, 1)], [QQ(-1, 1), QQ(0, 1)]]
    r = [[QQ(2, 1), QQ(0, 1), QQ(0, 1)]]
    assert dmp_ff_div(f, g, 1, QQ) == (q, r)
    f = dmp_normal([[1], [], [1, 0, 0]], 1, QQ)
    g = dmp_normal([[2], [-2, 0]], 1, QQ)
    q = [[QQ(1, 2)], [QQ(1, 2), QQ(0, 1)]]
    r = [[QQ(2, 1), QQ(0, 1), QQ(0, 1)]]
    assert dmp_ff_div(f, g, 1, QQ) == (q, r)