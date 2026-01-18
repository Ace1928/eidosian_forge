import inspect
import toolz
from toolz.functoolz import (thread_first, thread_last, memoize, curry,
from operator import add, mul, itemgetter
from toolz.utils import raises
from functools import partial
def test_memoize_key():

    @memoize(key=lambda args, kwargs: args[0])
    def f(x, y, *args, **kwargs):
        return x + y
    assert f(1, 2) == 3
    assert f(1, 3) == 3