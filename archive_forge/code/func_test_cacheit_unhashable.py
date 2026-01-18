import sys
from sympy.core.cache import cacheit, cached_property, lazy_function
from sympy.testing.pytest import raises
def test_cacheit_unhashable():

    @cacheit
    def testit(x):
        return x
    assert testit(1) == 1
    assert testit(1) == 1
    a = {}
    assert testit(a) == {}
    a[1] = 2
    assert testit(a) == {1: 2}