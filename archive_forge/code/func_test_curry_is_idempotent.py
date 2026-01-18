import inspect
import toolz
from toolz.functoolz import (thread_first, thread_last, memoize, curry,
from operator import add, mul, itemgetter
from toolz.utils import raises
from functools import partial
def test_curry_is_idempotent():

    def foo(a, b, c=1):
        return a + b + c
    f = curry(foo, 1, c=2)
    g = curry(f)
    assert isinstance(f, curry)
    assert isinstance(g, curry)
    assert not isinstance(g.func, curry)
    assert not hasattr(g.func, 'func')
    assert f.func == g.func
    assert f.args == g.args
    assert f.keywords == g.keywords