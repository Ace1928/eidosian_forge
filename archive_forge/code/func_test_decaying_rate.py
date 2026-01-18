from thinc.api import (
def test_decaying_rate():
    rates = decaying(0.001, 0.0001)
    rate = next(rates)
    assert rate == 0.001
    next_rate = next(rates)
    assert next_rate < rate
    assert next_rate > 0
    assert next_rate > next(rates)