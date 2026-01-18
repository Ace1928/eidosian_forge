from mpmath import *
def test_fsum():
    mp.dps = 15
    assert fsum([]) == 0
    assert fsum([-4]) == -4
    assert fsum([2, 3]) == 5
    assert fsum([1e-100, 1]) == 1
    assert fsum([1, 1e-100]) == 1
    assert fsum([1e+100, 1]) == 1e+100
    assert fsum([1, 1e+100]) == 1e+100
    assert fsum([1e-100, 0]) == 1e-100
    assert fsum([1e-100, 1e+100, 1e-100]) == 1e+100
    assert fsum([2, 1 + 1j, 1]) == 4 + 1j
    assert fsum([2, inf, 3]) == inf
    assert fsum([2, -1], absolute=1) == 3
    assert fsum([2, -1], squared=1) == 5
    assert fsum([1, 1 + j], squared=1) == 1 + 2j
    assert fsum([1, 3 + 4j], absolute=1) == 6
    assert fsum([1, 2 + 3j], absolute=1, squared=1) == 14
    assert isnan(fsum([inf, -inf]))
    assert fsum([inf, -inf], absolute=1) == inf
    assert fsum([inf, -inf], squared=1) == inf
    assert fsum([inf, -inf], absolute=1, squared=1) == inf
    assert iv.fsum([1, mpi(2, 3)]) == mpi(3, 4)