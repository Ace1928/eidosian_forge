import functools
import inspect
import itertools
import operator
import toolz
from toolz.functoolz import (curry, is_valid_args, is_partial_args, is_arity,
from toolz._signatures import builtins
import toolz._signatures as _sigs
from toolz.utils import raises
def test_is_arity():
    assert is_arity(0, lambda: None)
    assert is_arity(1, lambda: None) is False
    assert is_arity(1, lambda x: None)
    assert is_arity(3, lambda x, y, z: None)
    assert is_arity(1, lambda x, *args: None) is False
    assert is_arity(1, lambda x, **kwargs: None) is False
    assert is_arity(1, all)
    assert is_arity(2, map) is False
    assert is_arity(2, range) is None