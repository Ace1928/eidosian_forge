from sympy.multipledispatch.dispatcher import (Dispatcher, MDNotImplementedError,
from sympy.testing.pytest import raises, warns
def test_dispatch_method():
    f = Dispatcher('f')

    @f.register(list)
    def rev(x):
        return x[::-1]

    @f.register(int, int)
    def add(x, y):
        return x + y

    class MyList(list):
        pass
    assert f.dispatch(list) is rev
    assert f.dispatch(MyList) is rev
    assert f.dispatch(int, int) is add