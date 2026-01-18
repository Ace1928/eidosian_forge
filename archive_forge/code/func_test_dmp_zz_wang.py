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
def test_dmp_zz_wang():
    R, x, y, z = ring('x,y,z', ZZ)
    UV, _x = ring('x', ZZ)
    p = ZZ(nextprime(R.dmp_zz_mignotte_bound(w_1)))
    assert p == 6291469
    t_1, k_1, e_1 = (y, 1, ZZ(-14))
    t_2, k_2, e_2 = (z, 2, ZZ(3))
    t_3, k_3, e_3 = (y + z, 2, ZZ(-11))
    t_4, k_4, e_4 = (y - z, 1, ZZ(-17))
    T = [t_1, t_2, t_3, t_4]
    K = [k_1, k_2, k_3, k_4]
    E = [e_1, e_2, e_3, e_4]
    T = zip([t.drop(x) for t in T], K)
    A = [ZZ(-14), ZZ(3)]
    S = R.dmp_eval_tail(w_1, A)
    cs, s = UV.dup_primitive(S)
    assert cs == 1 and s == S == 1036728 * _x ** 6 + 915552 * _x ** 5 + 55748 * _x ** 4 + 105621 * _x ** 3 - 17304 * _x ** 2 - 26841 * _x - 644
    assert R.dmp_zz_wang_non_divisors(E, cs, ZZ(4)) == [7, 3, 11, 17]
    assert UV.dup_sqf_p(s) and UV.dup_degree(s) == R.dmp_degree(w_1)
    _, H = UV.dup_zz_factor_sqf(s)
    h_1 = 44 * _x ** 2 + 42 * _x + 1
    h_2 = 126 * _x ** 2 - 9 * _x + 28
    h_3 = 187 * _x ** 2 - 23
    assert H == [h_1, h_2, h_3]
    LC = [lc.drop(x) for lc in [-4 * y - 4 * z, -y * z ** 2, y ** 2 - z ** 2]]
    assert R.dmp_zz_wang_lead_coeffs(w_1, T, cs, E, H, A) == (w_1, H, LC)
    factors = R.dmp_zz_wang_hensel_lifting(w_1, H, LC, A, p)
    assert R.dmp_expand(factors) == w_1