from mpmath import *
from mpmath.libmp import round_up, from_float, mpf_zeta_int
def test_gamma_huge_1():
    mp.dps = 500
    x = mpf(10 ** 10) / 7
    mp.dps = 15
    assert str(gamma(x)) == '6.26075321389519e+12458010678'
    mp.dps = 50
    assert str(gamma(x)) == '6.2607532138951929201303779291707455874010420783933e+12458010678'
    mp.dps = 15