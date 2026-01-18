from mpmath import *
def test_interval_gamma():
    mp.dps = 15
    iv.dps = 15
    assert iv.rgamma(0) == 0
    assert iv.fac(0) == 1
    assert iv.fac(1) == 1
    assert iv.fac(2) == 2
    assert iv.fac(3) == 6
    assert iv.gamma(0) == [-inf, inf]
    assert iv.gamma(1) == 1
    assert iv.gamma(2) == 1
    assert iv.gamma(3) == 2
    assert -3.544907701811032 in iv.gamma(-0.5)
    assert iv.loggamma(1) == 0
    assert iv.loggamma(2) == 0
    assert 0.6931471805599453 in iv.loggamma(3)
    xs = [iv.mpc([2, 3], [1, 4]), iv.mpc([2, 3], [-4, -1]), iv.mpc([2, 3], [-1, 4]), iv.mpc([2, 3], [-4, 1]), iv.mpc([2, 3], [-4, 4]), iv.mpc([-3, -2], [2, 4]), iv.mpc([-3, -2], [-4, -2])]
    for x in xs:
        ys = [mp.loggamma(mp.mpc(x.a, x.c)), mp.loggamma(mp.mpc(x.b, x.c)), mp.loggamma(mp.mpc(x.a, x.d)), mp.loggamma(mp.mpc(x.b, x.d))]
        if 0 in x.imag:
            ys += [mp.loggamma(x.a), mp.loggamma(x.b)]
        min_real = min([y.real for y in ys])
        max_real = max([y.real for y in ys])
        min_imag = min([y.imag for y in ys])
        max_imag = max([y.imag for y in ys])
        z = iv.loggamma(x)
        assert z.a.ae(min_real)
        assert z.b.ae(max_real)
        assert z.c.ae(min_imag)
        assert z.d.ae(max_imag)