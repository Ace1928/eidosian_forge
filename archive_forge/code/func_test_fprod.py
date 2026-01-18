from mpmath import *
def test_fprod():
    mp.dps = 15
    assert fprod([]) == 1
    assert fprod([2, 3]) == 6