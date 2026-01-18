from datetime import datetime
from itertools import permutations
import numpy as np
from pandas._libs import algos as libalgos
import pandas._testing as tm
def test_infinity_sort(self):
    Inf = libalgos.Infinity()
    NegInf = libalgos.NegInfinity()
    ref_nums = [NegInf, float('-inf'), -1e+100, 0, 1e+100, float('inf'), Inf]
    assert all((Inf >= x for x in ref_nums))
    assert all((Inf > x or x is Inf for x in ref_nums))
    assert Inf >= Inf and Inf == Inf
    assert not Inf < Inf and (not Inf > Inf)
    assert libalgos.Infinity() == libalgos.Infinity()
    assert not libalgos.Infinity() != libalgos.Infinity()
    assert all((NegInf <= x for x in ref_nums))
    assert all((NegInf < x or x is NegInf for x in ref_nums))
    assert NegInf <= NegInf and NegInf == NegInf
    assert not NegInf < NegInf and (not NegInf > NegInf)
    assert libalgos.NegInfinity() == libalgos.NegInfinity()
    assert not libalgos.NegInfinity() != libalgos.NegInfinity()
    for perm in permutations(ref_nums):
        assert sorted(perm) == ref_nums
    np.array([libalgos.Infinity()] * 32).argsort()
    np.array([libalgos.NegInfinity()] * 32).argsort()