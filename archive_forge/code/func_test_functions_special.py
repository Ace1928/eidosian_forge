from mpmath import *
def test_functions_special():
    assert exp(inf) == inf
    assert exp(-inf) == 0
    assert isnan(exp(nan))
    assert log(inf) == inf
    assert isnan(log(nan))
    assert isnan(sin(inf))
    assert isnan(sin(nan))
    assert atan(inf).ae(pi / 2)
    assert atan(-inf).ae(-pi / 2)
    assert isnan(sqrt(nan))
    assert sqrt(inf) == inf