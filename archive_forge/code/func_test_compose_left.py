import inspect
import toolz
from toolz.functoolz import (thread_first, thread_last, memoize, curry,
from operator import add, mul, itemgetter
from toolz.utils import raises
from functools import partial
def test_compose_left():
    for compose_left_args, args, kw, expected in generate_compose_left_test_cases():
        assert compose_left(*compose_left_args)(*args, **kw) == expected