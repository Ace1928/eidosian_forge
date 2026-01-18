import math
import pytest
from mpmath import *
def test_hyper_2f2():
    mp.dps = 15
    assert hyper([1, 2], [3, 4], 5) == hyp2f2(1, 2, 3, 4, 5)
    a1, a2, b1, b2 = ((3, 10), 4, (1, 2), 1.0 / 16)
    assert hyp2f2(a1, a2, b1, b2, 10).ae(448225936.3377557)
    assert hyp2f2(a1, a2, b1, b2, 10000).ae('1.2012553712966636711e+4358')
    assert hyp2f2(a1, a2, b1, b2, -20000).ae(-0.04182343755661214)
    assert hyp2f2(a1, a2, b1, b2, 10 ** 20).ae('1.1148680024303263661e+43429448190325182840')