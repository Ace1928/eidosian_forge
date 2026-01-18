import functools
import inspect
import itertools
import operator
import toolz
from toolz.functoolz import (curry, is_valid_args, is_partial_args, is_arity,
from toolz._signatures import builtins
import toolz._signatures as _sigs
from toolz.utils import raises
def test_introspect_curry_valid_py3(check_valid=is_valid_args, incomplete=False):
    orig_check_valid = check_valid
    check_valid = lambda _func, *args, **kwargs: orig_check_valid(_func, args, kwargs)
    f = toolz.curry(make_func('x, y, z=0'))
    assert check_valid(f)
    assert check_valid(f, 1)
    assert check_valid(f, 1, 2)
    assert check_valid(f, 1, 2, 3)
    assert check_valid(f, 1, 2, 3, 4) is False
    assert check_valid(f, invalid_keyword=True) is False
    assert check_valid(f(1))
    assert check_valid(f(1), 2)
    assert check_valid(f(1), 2, 3)
    assert check_valid(f(1), 2, 3, 4) is False
    assert check_valid(f(1), x=2) is False
    assert check_valid(f(1), y=2)
    assert check_valid(f(x=1), 2) is False
    assert check_valid(f(x=1), y=2)
    assert check_valid(f(y=2), 1)
    assert check_valid(f(y=2), 1, z=3)
    assert check_valid(f(y=2), 1, 3) is False
    f = toolz.curry(make_func('x, y, z=0'), 1, x=1)
    assert check_valid(f) is False
    assert check_valid(f, z=3) is False
    f = toolz.curry(make_func('x, y, *args, z'))
    assert check_valid(f)
    assert check_valid(f, 0)
    assert check_valid(f(1), 0)
    assert check_valid(f(1, 2), 0)
    assert check_valid(f(1, 2, 3), 0)
    assert check_valid(f(1, 2, 3, 4), 0)
    assert check_valid(f(1, 2, 3, 4), z=4)
    assert check_valid(f(x=1))
    assert check_valid(f(x=1), 1) is False
    assert check_valid(f(x=1), y=2)