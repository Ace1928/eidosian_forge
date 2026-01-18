from sympy.polys.rings import ring, xring
from sympy.polys.domains import FF, ZZ, QQ, ZZ_I, QQ_I, RR, EX
from sympy.polys import polyconfig as config
from sympy.polys.polyerrors import DomainError
from sympy.polys.polyclasses import ANP
from sympy.polys.specialpolys import f_polys, w_polys
from sympy.core.numbers import I
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.ntheory.generate import nextprime
from sympy.testing.pytest import raises, XFAIL
def test_dup_factor_list():
    R, x = ring('x', ZZ)
    assert R.dup_factor_list(0) == (0, [])
    assert R.dup_factor_list(7) == (7, [])
    R, x = ring('x', QQ)
    assert R.dup_factor_list(0) == (0, [])
    assert R.dup_factor_list(QQ(1, 7)) == (QQ(1, 7), [])
    R, x = ring('x', ZZ['t'])
    assert R.dup_factor_list(0) == (0, [])
    assert R.dup_factor_list(7) == (7, [])
    R, x = ring('x', QQ['t'])
    assert R.dup_factor_list(0) == (0, [])
    assert R.dup_factor_list(QQ(1, 7)) == (QQ(1, 7), [])
    R, x = ring('x', ZZ)
    assert R.dup_factor_list_include(0) == [(0, 1)]
    assert R.dup_factor_list_include(7) == [(7, 1)]
    assert R.dup_factor_list(x ** 2 + 2 * x + 1) == (1, [(x + 1, 2)])
    assert R.dup_factor_list_include(x ** 2 + 2 * x + 1) == [(x + 1, 2)]
    assert R.dup_factor_list(6 * x ** 2 - 5 * x - 6) == (1, [(2 * x - 3, 1), (3 * x + 2, 1)])
    R, x = ring('x', QQ)
    assert R.dup_factor_list(QQ(1, 2) * x ** 2 + x + QQ(1, 2)) == (QQ(1, 2), [(x + 1, 2)])
    R, x = ring('x', FF(2))
    assert R.dup_factor_list(x ** 2 + 1) == (1, [(x + 1, 2)])
    R, x = ring('x', RR)
    assert R.dup_factor_list(1.0 * x ** 2 + 2.0 * x + 1.0) == (1.0, [(1.0 * x + 1.0, 2)])
    assert R.dup_factor_list(2.0 * x ** 2 + 4.0 * x + 2.0) == (2.0, [(1.0 * x + 1.0, 2)])
    f = 6.7225336055071 * x ** 2 - 10.6463972754741 * x - 0.33469524022264
    coeff, factors = R.dup_factor_list(f)
    assert coeff == RR(10.6463972754741)
    assert len(factors) == 1
    assert factors[0][0].max_norm() == RR(1.0)
    assert factors[0][1] == 1
    Rt, t = ring('t', ZZ)
    R, x = ring('x', Rt)
    f = 4 * t * x ** 2 + 4 * t ** 2 * x
    assert R.dup_factor_list(f) == (4 * t, [(x, 1), (x + t, 1)])
    Rt, t = ring('t', QQ)
    R, x = ring('x', Rt)
    f = QQ(1, 2) * t * x ** 2 + QQ(1, 2) * t ** 2 * x
    assert R.dup_factor_list(f) == (QQ(1, 2) * t, [(x, 1), (x + t, 1)])
    R, x = ring('x', QQ.algebraic_field(I))

    def anp(element):
        return ANP(element, [QQ(1), QQ(0), QQ(1)], QQ)
    f = anp([QQ(1, 1)]) * x ** 4 + anp([QQ(2, 1)]) * x ** 2
    assert R.dup_factor_list(f) == (anp([QQ(1, 1)]), [(anp([QQ(1, 1)]) * x, 2), (anp([QQ(1, 1)]) * x ** 2 + anp([]) * x + anp([QQ(2, 1)]), 1)])
    R, x = ring('x', EX)
    raises(DomainError, lambda: R.dup_factor_list(EX(sin(1))))