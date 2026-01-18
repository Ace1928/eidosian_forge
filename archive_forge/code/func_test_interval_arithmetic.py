from mpmath import *
def test_interval_arithmetic():
    iv.dps = 15
    assert mpi(2) + mpi(3, 4) == mpi(5, 6)
    assert mpi(1, 2) ** 2 == mpi(1, 4)
    assert mpi(1) + mpi(0, 1e-50) == mpi(1, mpf('1.0000000000000002'))
    x = 1 / (1 / mpi(3))
    assert x.a < 3 < x.b
    x = mpi(2) ** mpi(0.5)
    iv.dps += 5
    sq = iv.sqrt(2)
    iv.dps -= 5
    assert x.a < sq < x.b
    assert mpi(1) / mpi(1, inf)
    assert mpi(2, 3) / inf == mpi(0, 0)
    assert mpi(0) / inf == 0
    assert mpi(0) / 0 == mpi(-inf, inf)
    assert mpi(inf) / 0 == mpi(-inf, inf)
    assert mpi(0) * inf == mpi(-inf, inf)
    assert 1 / mpi(2, inf) == mpi(0, 0.5)
    assert str(mpi(50, 50) * mpi(-10, -10) / 3) == '[-166.66666666666668561, -166.66666666666665719]'
    assert mpi(0, 4) ** 3 == mpi(0, 64)
    assert mpi(2, 4).mid == 3
    iv.dps = 30
    a = mpi(iv.pi)
    iv.dps = 15
    b = +a
    assert b.a < a.a
    assert b.b > a.b
    a = mpi(iv.pi)
    assert a == +a
    assert abs(mpi(-1, 2)) == mpi(0, 2)
    assert abs(mpi(0.5, 2)) == mpi(0.5, 2)
    assert abs(mpi(-3, 2)) == mpi(0, 3)
    assert abs(mpi(-3, -0.5)) == mpi(0.5, 3)
    assert mpi(0) * mpi(2, 3) == mpi(0)
    assert mpi(2, 3) * mpi(0) == mpi(0)
    assert mpi(1, 3).delta == 2
    assert mpi(1, 2) - mpi(3, 4) == mpi(-3, -1)
    assert mpi(-inf, 0) - mpi(0, inf) == mpi(-inf, 0)
    assert mpi(-inf, 0) - mpi(-inf, inf) == mpi(-inf, inf)
    assert mpi(0, inf) - mpi(-inf, 1) == mpi(-1, inf)