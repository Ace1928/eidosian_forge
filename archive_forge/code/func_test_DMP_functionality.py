from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.domains import ZZ, QQ
from sympy.polys.polyclasses import DMP, DMF, ANP
from sympy.polys.polyerrors import (CoercionFailed, ExactQuotientFailed,
from sympy.polys.specialpolys import f_polys
from sympy.testing.pytest import raises
def test_DMP_functionality():
    f = DMP([[1], [2, 0], [1, 0, 0]], ZZ)
    g = DMP([[1], [1, 0]], ZZ)
    h = DMP([[1]], ZZ)
    assert f.degree() == 2
    assert f.degree_list() == (2, 2)
    assert f.total_degree() == 2
    assert f.LC() == ZZ(1)
    assert f.TC() == ZZ(0)
    assert f.nth(1, 1) == ZZ(2)
    raises(TypeError, lambda: f.nth(0, 'x'))
    assert f.max_norm() == 2
    assert f.l1_norm() == 4
    u = DMP([[2], [2, 0]], ZZ)
    assert f.diff(m=1, j=0) == u
    assert f.diff(m=1, j=1) == u
    raises(TypeError, lambda: f.diff(m='x', j=0))
    u = DMP([1, 2, 1], ZZ)
    v = DMP([1, 2, 1], ZZ)
    assert f.eval(a=1, j=0) == u
    assert f.eval(a=1, j=1) == v
    assert f.eval(1).eval(1) == ZZ(4)
    assert f.cofactors(g) == (g, g, h)
    assert f.gcd(g) == g
    assert f.lcm(g) == f
    u = DMP([[QQ(45), QQ(30), QQ(5)]], QQ)
    v = DMP([[QQ(1), QQ(2, 3), QQ(1, 9)]], QQ)
    assert u.monic() == v
    assert (4 * f).content() == ZZ(4)
    assert (4 * f).primitive() == (ZZ(4), f)
    f = DMP([[1], [2], [3], [4], [5], [6]], ZZ)
    assert f.trunc(3) == DMP([[1], [-1], [], [1], [-1], []], ZZ)
    f = DMP(f_4, ZZ)
    assert f.sqf_part() == -f
    assert f.sqf_list() == (ZZ(-1), [(-f, 1)])
    f = DMP([[-1], [], [], [5]], ZZ)
    g = DMP([[3, 1], [], []], ZZ)
    h = DMP([[45, 30, 5]], ZZ)
    r = DMP([675, 675, 225, 25], ZZ)
    assert f.subresultants(g) == [f, g, h]
    assert f.resultant(g) == r
    f = DMP([1, 3, 9, -13], ZZ)
    assert f.discriminant() == -11664
    f = DMP([QQ(2), QQ(0)], QQ)
    g = DMP([QQ(1), QQ(0), QQ(-16)], QQ)
    s = DMP([QQ(1, 32), QQ(0)], QQ)
    t = DMP([QQ(-1, 16)], QQ)
    h = DMP([QQ(1)], QQ)
    assert f.half_gcdex(g) == (s, h)
    assert f.gcdex(g) == (s, t, h)
    assert f.invert(g) == s
    f = DMP([[1], [2], [3]], QQ)
    raises(ValueError, lambda: f.half_gcdex(f))
    raises(ValueError, lambda: f.gcdex(f))
    raises(ValueError, lambda: f.invert(f))
    f = DMP([1, 0, 20, 0, 150, 0, 500, 0, 625, -2, 0, -10, 9], ZZ)
    g = DMP([1, 0, 0, -2, 9], ZZ)
    h = DMP([1, 0, 5, 0], ZZ)
    assert g.compose(h) == f
    assert f.decompose() == [g, h]
    f = DMP([[1], [2], [3]], QQ)
    raises(ValueError, lambda: f.decompose())
    raises(ValueError, lambda: f.sturm())