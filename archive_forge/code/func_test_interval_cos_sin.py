from mpmath import *
def test_interval_cos_sin():
    iv.dps = 15
    cos = iv.cos
    sin = iv.sin
    tan = iv.tan
    pi = iv.pi
    assert cos(mpi(0)) == 1
    assert sin(mpi(0)) == 0
    assert cos(mpi(0, 1)) == mpi(0.5403023058681397, 1.0)
    assert sin(mpi(0, 1)) == mpi(0, 0.8414709848078966)
    assert cos(mpi(1, 2)) == mpi(-0.4161468365471424, 0.5403023058681398)
    assert sin(mpi(1, 2)) == mpi(0.8414709848078965, 1.0)
    assert sin(mpi(1, 2.5)) == mpi(0.5984721441039564, 1.0)
    assert cos(mpi(-1, 1)) == mpi(0.5403023058681397, 1.0)
    assert cos(mpi(-1, 0.5)) == mpi(0.5403023058681397, 1.0)
    assert cos(mpi(-1, 1.5)) == mpi(0.0707372016677029, 1.0)
    assert sin(mpi(-1, 1)) == mpi(-0.8414709848078966, 0.8414709848078966)
    assert sin(mpi(-1, 0.5)) == mpi(-0.8414709848078966, 0.479425538604203)
    assert mpi(-0.8414709848078966, 1e-100) in sin(mpi(-1, 1e-100))
    assert mpi(-2e-100, 1e-100) in sin(mpi(-2e-100, 1e-100))
    assert cos(mpi(2, 2.5))
    assert cos(mpi(3.5, 4)) == mpi(-0.9364566872907963, -0.6536436208636118)
    assert cos(mpi(5, 5.5)) == mpi(0.28366218546322625, 0.7086697742912601)
    assert mpi(0.5984721441039565, 0.9092974268256817) in sin(mpi(2, 2.5))
    assert sin(mpi(3.5, 4)) == mpi(-0.7568024953079283, -0.35078322768961984)
    assert sin(mpi(5, 5.5)) == mpi(-0.9589242746631386, -0.7055403255703918)
    iv.dps = 55
    w = 4 * 10 ** 50 + mpi(0.5)
    for p in [15, 40, 80]:
        iv.dps = p
        assert 0 in sin(4 * mpi(pi))
        assert 0 in sin(4 * 10 ** 50 * mpi(pi))
        assert 0 in cos((4 + 0.5) * mpi(pi))
        assert 0 in cos(w * mpi(pi))
        assert 1 in cos(4 * mpi(pi))
        assert 1 in cos(4 * 10 ** 50 * mpi(pi))
    iv.dps = 15
    assert cos(mpi(2, inf)) == mpi(-1, 1)
    assert sin(mpi(2, inf)) == mpi(-1, 1)
    assert cos(mpi(-inf, 2)) == mpi(-1, 1)
    assert sin(mpi(-inf, 2)) == mpi(-1, 1)
    u = tan(mpi(0.5, 1))
    assert mpf(u.a).ae(mp.tan(0.5))
    assert mpf(u.b).ae(mp.tan(1))
    v = iv.cot(mpi(0.5, 1))
    assert mpf(v.a).ae(mp.cot(1))
    assert mpf(v.b).ae(mp.cot(0.5))
    for n in range(-5, 7, 2):
        x = iv.cos(n * iv.pi)
        assert -1 in x
        assert x >= -1
        assert x != -1
        x = iv.sin((n + 0.5) * iv.pi)
        assert -1 in x
        assert x >= -1
        assert x != -1
    for n in range(-6, 8, 2):
        x = iv.cos(n * iv.pi)
        assert 1 in x
        assert x <= 1
        if n:
            assert x != 1
        x = iv.sin((n + 0.5) * iv.pi)
        assert 1 in x
        assert x <= 1
        assert x != 1
    for n in range(-6, 7):
        x = iv.cos((n + 0.5) * iv.pi)
        assert x.a < 0 < x.b
        x = iv.sin(n * iv.pi)
        if n:
            assert x.a < 0 < x.b