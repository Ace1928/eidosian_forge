import functools
import inspect
import itertools
import operator
import toolz
from toolz.functoolz import (curry, is_valid_args, is_partial_args, is_arity,
from toolz._signatures import builtins
import toolz._signatures as _sigs
from toolz.utils import raises
def test_func_keyword():

    def f(func=None):
        pass
    assert is_valid_args(f, (), {})
    assert is_valid_args(f, (None,), {})
    assert is_valid_args(f, (), {'func': None})
    assert is_valid_args(f, (None,), {'func': None}) is False
    assert is_partial_args(f, (), {})
    assert is_partial_args(f, (None,), {})
    assert is_partial_args(f, (), {'func': None})
    assert is_partial_args(f, (None,), {'func': None}) is False