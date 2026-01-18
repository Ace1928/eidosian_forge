from sympy.external import import_module
from sympy.plotting.intervalmath import (
def test_imin():
    a = imin(interval(1, 3), interval(2, 5), interval(-1, 3))
    assert a.start == -1
    assert a.end == 3
    a = imin(-2, interval(1, 4))
    assert a.start == -2
    assert a.end == -2
    a = imin(5, interval(3, 4), interval(-2, 2, is_valid=False))
    assert a.start == 3
    assert a.end == 4