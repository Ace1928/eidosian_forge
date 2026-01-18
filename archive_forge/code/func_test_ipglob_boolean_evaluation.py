from netaddr import (
def test_ipglob_boolean_evaluation():
    assert bool(IPGlob('*.*.*.*'))
    assert bool(IPGlob('0.0.0.0'))