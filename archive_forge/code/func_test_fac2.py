from mpmath import *
from mpmath.libmp import round_up, from_float, mpf_zeta_int
def test_fac2():
    mp.dps = 15
    assert [fac2(n) for n in range(10)] == [1, 1, 2, 3, 8, 15, 48, 105, 384, 945]
    assert fac2(-5).ae(1.0 / 3)
    assert fac2(-11).ae(-1.0 / 945)
    assert fac2(50).ae(5.204698426366666e+32)
    assert fac2(0.5 + 0.75j).ae(0.8154676939468807 - 0.34901016085573267j)
    assert fac2(inf) == inf
    assert isnan(fac2(-inf))