from sympy.multipledispatch.dispatcher import (Dispatcher, MDNotImplementedError,
from sympy.testing.pytest import raises, warns
def test_dispatcher_as_decorator():
    f = Dispatcher('f')

    @f.register(int)
    def inc(x):
        return x + 1

    @f.register(float)
    def inc(x):
        return x - 1
    assert f(1) == 2
    assert f(1.0) == 0.0