import inspect
import toolz
from toolz.functoolz import (thread_first, thread_last, memoize, curry,
from operator import add, mul, itemgetter
from toolz.utils import raises
from functools import partial
def test_curry_bind():

    @curry
    def add(x=1, y=2):
        return x + y
    assert add() == add(1, 2)
    assert add.bind(10)(20) == add(10, 20)
    assert add.bind(10).bind(20)() == add(10, 20)
    assert add.bind(x=10)(y=20) == add(10, 20)
    assert add.bind(x=10).bind(y=20)() == add(10, 20)