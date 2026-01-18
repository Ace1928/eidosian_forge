import inspect
import toolz
from toolz.functoolz import (thread_first, thread_last, memoize, curry,
from operator import add, mul, itemgetter
from toolz.utils import raises
from functools import partial
def generate_compose_left_test_cases():
    """
    Generate test cases for parametrized tests of the compose function.

    These are based on, and equivalent to, those produced by
    enerate_compose_test_cases().
    """
    return tuple(((tuple(reversed(compose_args)), args, kwargs, expected) for compose_args, args, kwargs, expected in generate_compose_test_cases()))