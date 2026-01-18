import functools
import inspect
import itertools
import operator
import toolz
from toolz.functoolz import (curry, is_valid_args, is_partial_args, is_arity,
from toolz._signatures import builtins
import toolz._signatures as _sigs
from toolz.utils import raises
def test_introspect_curry_py3():
    f = toolz.curry(make_func(''))
    assert num_required_args(f) == 0
    assert is_arity(0, f)
    assert has_varargs(f) is False
    assert has_keywords(f) is False
    f = toolz.curry(make_func('x'))
    assert num_required_args(f) == 0
    assert is_arity(0, f) is False
    assert is_arity(1, f) is False
    assert has_varargs(f) is False
    assert has_keywords(f)
    f = toolz.curry(make_func('x, y, z=0'))
    assert num_required_args(f) == 0
    assert is_arity(0, f) is False
    assert is_arity(1, f) is False
    assert is_arity(2, f) is False
    assert is_arity(3, f) is False
    assert has_varargs(f) is False
    assert has_keywords(f)
    f = toolz.curry(make_func('*args, **kwargs'))
    assert num_required_args(f) == 0
    assert has_varargs(f)
    assert has_keywords(f)