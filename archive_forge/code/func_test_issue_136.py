from mpmath.libmp import *
from mpmath import *
import random
import time
import math
import cmath
def test_issue_136():
    for dps in [20, 80]:
        mp.dps = dps
        r = nthroot(mpf('-1e-20'), 4)
        assert r.ae(mpf(10) ** (-5) * (1 + j) * mpf(2) ** (-0.5))
    mp.dps = 80
    assert nthroot('-1e-3', 4).ae(mpf(10) ** (-3.0 / 4) * (1 + j) / sqrt(2))
    assert nthroot('-1e-6', 4).ae((1 + j) / (10 * sqrt(20)))
    mp.dps = 20
    assert nthroot('-1e100000000', 4).ae((1 + j) * mpf('1e25000000') / sqrt(2))
    mp.dps = 15