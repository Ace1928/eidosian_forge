from mpmath import *
def test_convert_special():
    float_inf = 1e+300 * 1e+300
    float_ninf = -float_inf
    float_nan = float_inf / float_ninf
    assert mpf(3) * float_inf == inf
    assert mpf(3) * float_ninf == -inf
    assert isnan(mpf(3) * float_nan)
    assert not mpf(3) < float_nan
    assert not mpf(3) > float_nan
    assert not mpf(3) <= float_nan
    assert not mpf(3) >= float_nan
    assert float(mpf('1e1000')) == float_inf
    assert float(mpf('-1e1000')) == float_ninf
    assert float(mpf('1e100000000000000000')) == float_inf
    assert float(mpf('-1e100000000000000000')) == float_ninf
    assert float(mpf('1e-100000000000000000')) == 0.0