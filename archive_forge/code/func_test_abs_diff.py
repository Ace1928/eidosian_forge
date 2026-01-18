from symengine import (
from symengine.test_utilities import raises
import unittest
def test_abs_diff():
    x = Symbol('x')
    y = Symbol('y')
    e = abs(x)
    assert e.diff(x) != e
    assert e.diff(x) != 0
    assert e.diff(y) == 0