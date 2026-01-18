import functools
import inspect
import itertools
import operator
import toolz
from toolz.functoolz import (curry, is_valid_args, is_partial_args, is_arity,
from toolz._signatures import builtins
import toolz._signatures as _sigs
from toolz.utils import raises
def test_has_varargs():
    assert has_varargs(lambda: None) is False
    assert has_varargs(lambda *args: None)
    assert has_varargs(lambda **kwargs: None) is False
    assert has_varargs(map)
    assert has_varargs(max) is None