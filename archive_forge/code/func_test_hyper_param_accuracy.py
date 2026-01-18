import math
import pytest
from mpmath import *
def test_hyper_param_accuracy():
    mp.dps = 15
    As = [n + 1e-10 for n in range(-5, -1)]
    Bs = [n + 1e-10 for n in range(-12, -5)]
    assert hyper(As, Bs, 10).ae(-381757055858.65265)
    assert legenp(0.5, 100, 0.25).ae(-2.412457656721131e+144)
    assert (hyp1f1(1000, 1, -100) * 10 ** 24).ae(5.258944543737017)
    assert (hyp2f1(10, -900, 10.5, 0.99) * 10 ** 24).ae(1.9185370579660768)
    assert (hyp2f1(1000, 1.5, -3.5, -1.5) * 10 ** 385).ae(-2.7367529051334)
    assert hyp2f1(-5, 10, 3, 0.5, zeroprec=500) == 0
    assert (hyp1f1(-10000, 1000, 100) * 10 ** 424).ae(-3.104608051582486)
    assert (hyp2f1(1000, 1.5, -3.5, -0.75, maxterms=100000) * 10 ** 231).ae(-4.0534790813914)
    assert legenp(2, 3, 0.25) == 0
    pytest.raises(ValueError, lambda: hypercomb(lambda a: [([], [], [], [], [a], [-a], 0.5)], [3]))
    assert hypercomb(lambda a: [([], [], [], [], [a], [-a], 0.5)], [3], infprec=200) == inf
    assert meijerg([[], []], [[0, 0, 0, 0], []], 0.1).ae(1.5680822343832352)
    assert (besselk(400, 400) * 10 ** 94).ae(1.4387057277018551)
    mp.dps = 5
    (hyp1f1(-5000.5, 1500, 100) * 10 ** 185).ae(8.518522967338194)
    (hyp1f1(-5000, 1500, 100) * 10 ** 185).ae(9.150121342456394)
    mp.dps = 15
    (hyp1f1(-5000.5, 1500, 100) * 10 ** 185).ae(8.518522967338194)
    (hyp1f1(-5000, 1500, 100) * 10 ** 185).ae(9.150121342456394)
    assert hyp0f1(fadd(-20, '1e-100', exact=True), 0.25).ae(1.8501442904010278e+49)
    assert hyp0f1((-20 * 10 ** 100 + 1, 10 ** 100), 0.25).ae(1.8501442904010278e+49)