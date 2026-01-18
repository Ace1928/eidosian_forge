from mpmath import *
def test_interval_identity():
    iv.dps = 15
    assert mpi(2) == mpi(2, 2)
    assert mpi(2) != mpi(-2, 2)
    assert not mpi(2) != mpi(2, 2)
    assert mpi(-1, 1) == mpi(-1, 1)
    assert str(mpi('0.1')) == '[0.099999999999999991673, 0.10000000000000000555]'
    assert repr(mpi('0.1')) == "mpi('0.099999999999999992', '0.10000000000000001')"
    u = mpi(-1, 3)
    assert -1 in u
    assert 2 in u
    assert 3 in u
    assert -1.1 not in u
    assert 3.1 not in u
    assert mpi(-1, 3) in u
    assert mpi(0, 1) in u
    assert mpi(-1.1, 2) not in u
    assert mpi(2.5, 3.1) not in u
    w = mpi(-inf, inf)
    assert mpi(-5, 5) in w
    assert mpi(2, inf) in w
    assert mpi(0, 2) in mpi(0, 10)
    assert not 3 in mpi(-inf, 0)