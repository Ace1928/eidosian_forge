import math
import pytest
from mpmath import *
def test_hyper_misc():
    mp.dps = 15
    assert hyp0f1(1, 0) == 1
    assert hyp1f1(1, 2, 0) == 1
    assert hyp1f2(1, 2, 3, 0) == 1
    assert hyp2f1(1, 2, 3, 0) == 1
    assert hyp2f2(1, 2, 3, 4, 0) == 1
    assert hyp2f3(1, 2, 3, 4, 5, 0) == 1
    assert hyper([], [], 0) == 1
    assert hyper([], [], -2).ae(exp(-2))
    assert hyper([2], [], 1.5) == 4
    assert hyp2f1((1, 3), (2, 3), (5, 6), mpf(27) / 32).ae(1.6)
    assert hyp2f1((1, 4), (1, 2), (3, 4), mpf(80) / 81).ae(1.8)
    assert hyp2f1((2, 3), (1, 1), (3, 2), (2 + j) / 3).ae(1.327531603558679 + 0.43958508009276925j)
    mp.dps = 25
    v = mpc('1.2282306665029814734863026', '-0.1225033830118305184672133')
    assert hyper([(3, 4), 2 + j, 1], [1, 5, j / 3], mpf(1) / 5 + j / 8).ae(v)
    mp.dps = 15