from mpmath import *
from mpmath.libmp import round_up, from_float, mpf_zeta_int
def test_barnesg():
    mp.dps = 15
    assert barnesg(0) == barnesg(-1) == 0
    assert [superfac(i) for i in range(8)] == [1, 1, 2, 12, 288, 34560, 24883200, 125411328000]
    assert str(superfac(1000)) == '3.24570818422368e+1177245'
    assert isnan(barnesg(nan))
    assert isnan(superfac(nan))
    assert isnan(hyperfac(nan))
    assert barnesg(inf) == inf
    assert superfac(inf) == inf
    assert hyperfac(inf) == inf
    assert isnan(superfac(-inf))
    assert barnesg(0.7).ae(0.8068722730141471)
    assert barnesg(2 + 3j).ae(-0.1781021386408217 + 0.04504542715447838j)
    assert [hyperfac(n) for n in range(7)] == [1, 1, 4, 108, 27648, 86400000, 4031078400000]
    assert [hyperfac(n) for n in range(0, -7, -1)] == [1, 1, -1, -4, 108, 27648, -86400000]
    a = barnesg(-3 + 0j)
    assert a == 0 and isinstance(a, mpc)
    a = hyperfac(-3 + 0j)
    assert a == -4 and isinstance(a, mpc)