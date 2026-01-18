from sympy.plotting.intervalmath import interval
from sympy.testing.pytest import raises
def test_interval_add():
    assert (interval(1, 2) + interval(2, 3) == interval(3, 5)) == (True, True)
    assert (1 + interval(1, 2) == interval(2, 3)) == (True, True)
    assert (interval(1, 2) + 1 == interval(2, 3)) == (True, True)
    compare = 1 + interval(0, float('inf')) == interval(1, float('inf'))
    assert compare == (True, True)
    a = 1 + interval(2, 5, is_valid=False)
    assert a.is_valid is False
    a = 1 + interval(2, 5, is_valid=None)
    assert a.is_valid is None
    a = interval(2, 5, is_valid=False) + interval(3, 5, is_valid=None)
    assert a.is_valid is False
    a = interval(3, 5) + interval(-1, 1, is_valid=None)
    assert a.is_valid is None
    a = interval(2, 5, is_valid=False) + 1
    assert a.is_valid is False