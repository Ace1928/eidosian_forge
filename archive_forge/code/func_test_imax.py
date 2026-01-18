from sympy.external import import_module
from sympy.plotting.intervalmath import (
def test_imax():
    a = imax(interval(-2, 2), interval(2, 7), interval(-3, 9))
    assert a.start == 2
    assert a.end == 9
    a = imax(8, interval(1, 4))
    assert a.start == 8
    assert a.end == 8
    a = imax(interval(1, 2), interval(3, 4), interval(-2, 2, is_valid=False))
    assert a.start == 3
    assert a.end == 4