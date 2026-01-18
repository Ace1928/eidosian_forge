from mpmath import *
def test_diffs():
    mp.dps = 15
    assert [chop(d) for d in diffs(sin, 0, 1)] == [0, 1]
    assert [chop(d) for d in diffs(sin, 0, 1, method='quad')] == [0, 1]
    assert [chop(d) for d in diffs(sin, 0, 2)] == [0, 1, 0]
    assert [chop(d) for d in diffs(sin, 0, 2, method='quad')] == [0, 1, 0]