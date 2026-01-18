import functools
import inspect
import itertools
import operator
import toolz
from toolz.functoolz import (curry, is_valid_args, is_partial_args, is_arity,
from toolz._signatures import builtins
import toolz._signatures as _sigs
from toolz.utils import raises
def test_is_valid(check_valid=is_valid_args, incomplete=False):
    orig_check_valid = check_valid
    check_valid = lambda func, *args, **kwargs: orig_check_valid(func, args, kwargs)
    f = make_func('')
    assert check_valid(f)
    assert check_valid(f, 1) is False
    assert check_valid(f, x=1) is False
    f = make_func('x')
    assert check_valid(f) is incomplete
    assert check_valid(f, 1)
    assert check_valid(f, x=1)
    assert check_valid(f, 1, x=2) is False
    assert check_valid(f, 1, y=2) is False
    assert check_valid(f, 1, 2) is False
    assert check_valid(f, x=1, y=2) is False
    f = make_func('x=1')
    assert check_valid(f)
    assert check_valid(f, 1)
    assert check_valid(f, x=1)
    assert check_valid(f, 1, x=2) is False
    assert check_valid(f, 1, y=2) is False
    assert check_valid(f, 1, 2) is False
    assert check_valid(f, x=1, y=2) is False
    f = make_func('*args')
    assert check_valid(f)
    assert check_valid(f, 1)
    assert check_valid(f, 1, 2)
    assert check_valid(f, x=1) is False
    f = make_func('**kwargs')
    assert check_valid(f)
    assert check_valid(f, x=1)
    assert check_valid(f, x=1, y=2)
    assert check_valid(f, 1) is False
    f = make_func('x, *args')
    assert check_valid(f) is incomplete
    assert check_valid(f, 1)
    assert check_valid(f, 1, 2)
    assert check_valid(f, x=1)
    assert check_valid(f, 1, x=1) is False
    assert check_valid(f, 1, y=1) is False
    f = make_func('x, y=1, **kwargs')
    assert check_valid(f) is incomplete
    assert check_valid(f, 1)
    assert check_valid(f, x=1)
    assert check_valid(f, 1, 2)
    assert check_valid(f, x=1, y=2, z=3)
    assert check_valid(f, 1, 2, y=3) is False
    f = make_func('a, b, c=3, d=4')
    assert check_valid(f) is incomplete
    assert check_valid(f, 1) is incomplete
    assert check_valid(f, 1, 2)
    assert check_valid(f, 1, c=3) is incomplete
    assert check_valid(f, 1, e=3) is False
    assert check_valid(f, 1, 2, e=3) is False
    assert check_valid(f, 1, 2, b=3) is False
    assert check_valid(1) is False