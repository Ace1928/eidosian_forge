from sympy.plotting.intervalmath import interval
from sympy.testing.pytest import raises
def test_interval_div():
    div = interval(1, 2, is_valid=False) / 3
    assert div == interval(-float('inf'), float('inf'), is_valid=False)
    div = interval(1, 2, is_valid=None) / 3
    assert div == interval(-float('inf'), float('inf'), is_valid=None)
    div = 3 / interval(1, 2, is_valid=None)
    assert div == interval(-float('inf'), float('inf'), is_valid=None)
    a = interval(1, 2) / 0
    assert a.is_valid is False
    a = interval(0.5, 1) / interval(-1, 0)
    assert a.is_valid is None
    a = interval(0, 1) / interval(0, 1)
    assert a.is_valid is None
    a = interval(-1, 1) / interval(-1, 1)
    assert a.is_valid is None
    a = interval(-1, 2) / interval(0.5, 1) == interval(-2.0, 4.0)
    assert a == (True, True)
    a = interval(0, 1) / interval(0.5, 1) == interval(0.0, 2.0)
    assert a == (True, True)
    a = interval(-1, 0) / interval(0.5, 1) == interval(-2.0, 0.0)
    assert a == (True, True)
    a = interval(-0.5, -0.25) / interval(0.5, 1) == interval(-1.0, -0.25)
    assert a == (True, True)
    a = interval(0.5, 1) / interval(0.5, 1) == interval(0.5, 2.0)
    assert a == (True, True)
    a = interval(0.5, 4) / interval(0.5, 1) == interval(0.5, 8.0)
    assert a == (True, True)
    a = interval(-1, -0.5) / interval(0.5, 1) == interval(-2.0, -0.5)
    assert a == (True, True)
    a = interval(-4, -0.5) / interval(0.5, 1) == interval(-8.0, -0.5)
    assert a == (True, True)
    a = interval(-1, 2) / interval(-2, -0.5) == interval(-4.0, 2.0)
    assert a == (True, True)
    a = interval(0, 1) / interval(-2, -0.5) == interval(-2.0, 0.0)
    assert a == (True, True)
    a = interval(-1, 0) / interval(-2, -0.5) == interval(0.0, 2.0)
    assert a == (True, True)
    a = interval(-0.5, -0.25) / interval(-2, -0.5) == interval(0.125, 1.0)
    assert a == (True, True)
    a = interval(0.5, 1) / interval(-2, -0.5) == interval(-2.0, -0.25)
    assert a == (True, True)
    a = interval(0.5, 4) / interval(-2, -0.5) == interval(-8.0, -0.25)
    assert a == (True, True)
    a = interval(-1, -0.5) / interval(-2, -0.5) == interval(0.25, 2.0)
    assert a == (True, True)
    a = interval(-4, -0.5) / interval(-2, -0.5) == interval(0.25, 8.0)
    assert a == (True, True)
    a = interval(-5, 5, is_valid=False) / 2
    assert a.is_valid is False