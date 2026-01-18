from sympy.multipledispatch.dispatcher import (Dispatcher, MDNotImplementedError,
from sympy.testing.pytest import raises, warns
def test_dispatcher():
    f = Dispatcher('f')
    f.add((int,), inc)
    f.add((float,), dec)
    with warns(DeprecationWarning, test_stacklevel=False):
        assert f.resolve((int,)) == inc
    assert f.dispatch(int) is inc
    assert f(1) == 2
    assert f(1.0) == 0.0