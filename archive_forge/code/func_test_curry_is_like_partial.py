import inspect
import toolz
from toolz.functoolz import (thread_first, thread_last, memoize, curry,
from operator import add, mul, itemgetter
from toolz.utils import raises
from functools import partial
def test_curry_is_like_partial():

    def foo(a, b, c=1):
        return a + b + c
    p, c = (partial(foo, 1, c=2), curry(foo)(1, c=2))
    assert p.keywords == c.keywords
    assert p.args == c.args
    assert p(3) == c(3)
    p, c = (partial(foo, 1), curry(foo)(1))
    assert p.keywords == c.keywords
    assert p.args == c.args
    assert p(3) == c(3)
    assert p(3, c=2) == c(3, c=2)
    p, c = (partial(foo, c=1), curry(foo)(c=1))
    assert p.keywords == c.keywords
    assert p.args == c.args
    assert p(1, 2) == c(1, 2)