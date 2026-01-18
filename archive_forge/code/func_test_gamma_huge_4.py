from mpmath import *
from mpmath.libmp import round_up, from_float, mpf_zeta_int
def test_gamma_huge_4():
    x = 3200 + 11500j
    mp.dps = 15
    assert str(gamma(x)) == '(8.95783268539713e+5164 - 1.94678798329735e+5164j)'
    mp.dps = 50
    assert str(gamma(x)) == '(8.9578326853971339570292952697675570822206567327092e+5164 - 1.9467879832973509568895402139429643650329524144794e+5164j)'
    mp.dps = 15