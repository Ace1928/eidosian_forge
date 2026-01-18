import sys
from sympy.core.cache import cacheit, cached_property, lazy_function
from sympy.testing.pytest import raises
def test_cached_property():

    class A:

        def __init__(self, value):
            self.value = value
            self.calls = 0

        @cached_property
        def prop(self):
            self.calls = self.calls + 1
            return self.value
    a = A(2)
    assert a.calls == 0
    assert a.prop == 2
    assert a.calls == 1
    assert a.prop == 2
    assert a.calls == 1
    b = A(None)
    assert b.prop == None