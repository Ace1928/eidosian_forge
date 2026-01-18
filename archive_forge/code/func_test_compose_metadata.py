import inspect
import toolz
from toolz.functoolz import (thread_first, thread_last, memoize, curry,
from operator import add, mul, itemgetter
from toolz.utils import raises
from functools import partial
def test_compose_metadata():

    def f(a):
        return a

    def g(a):
        return a
    composed = compose(f, g)
    assert composed.__name__ == 'f_of_g'
    assert composed.__doc__ == 'lambda *args, **kwargs: f(g(*args, **kwargs))'
    h = object()
    composed = compose(f, h)
    assert composed.__name__ == 'Compose'
    assert composed.__doc__ == 'A composition of functions'
    assert repr(composed) == 'Compose({!r}, {!r})'.format(f, h)
    assert composed == compose(f, h)
    assert composed == AlwaysEquals()
    assert not composed == compose(h, f)
    assert not composed == object()
    assert not composed == NeverEquals()
    assert composed != compose(h, f)
    assert composed != NeverEquals()
    assert composed != object()
    assert not composed != compose(f, h)
    assert not composed != AlwaysEquals()
    assert hash(composed) == hash(compose(f, h))
    assert hash(composed) != hash(compose(h, f))
    bindable = compose(str, lambda x: x * 2, lambda x, y=0: int(x) + y)

    class MyClass:

        def __int__(self):
            return 8
        my_method = bindable
        my_static_method = staticmethod(bindable)
    assert MyClass.my_method(3) == '6'
    assert MyClass.my_method(3, y=2) == '10'
    assert MyClass.my_static_method(2) == '4'
    assert MyClass().my_method() == '16'
    assert MyClass().my_method(y=3) == '22'
    assert MyClass().my_static_method(0) == '0'
    assert MyClass().my_static_method(0, 1) == '2'
    assert compose(f, h).__wrapped__ is h
    if hasattr(toolz, 'sandbox'):
        assert compose(f, h).__class__.__wrapped__ is None

    def myfunc(a, b, c, *d, **e):
        return 4

    def otherfunc(f):
        return 'result: {}'.format(f)
    myfunc.__annotations__ = {'a': int, 'b': str, 'c': float, 'd': int, 'e': bool, 'return': int}
    otherfunc.__annotations__ = {'f': int, 'return': str}
    composed = compose(otherfunc, myfunc)
    sig = inspect.signature(composed)
    assert sig.parameters == inspect.signature(myfunc).parameters
    assert sig.return_annotation == str

    class MyClass:
        method = composed
    assert len(inspect.signature(MyClass().method).parameters) == 4