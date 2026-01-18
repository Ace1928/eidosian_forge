import functools
import inspect
import itertools
import operator
import toolz
from toolz.functoolz import (curry, is_valid_args, is_partial_args, is_arity,
from toolz._signatures import builtins
import toolz._signatures as _sigs
from toolz.utils import raises
def test_has_unknown_args():
    assert has_varargs(1) is False
    assert has_varargs(map)
    assert has_varargs(make_func('')) is False
    assert has_varargs(make_func('x, y, z')) is False
    assert has_varargs(make_func('*args'))
    assert has_varargs(make_func('**kwargs')) is False
    assert has_varargs(make_func('x, y, *args, **kwargs'))
    assert has_varargs(make_func('x, y, z=1')) is False
    assert has_varargs(make_func('x, y, z=1, **kwargs')) is False
    f = make_func('*args')
    f.__signature__ = 34
    assert has_varargs(f) is False

    class RaisesValueError(object):

        def __call__(self):
            pass

        @property
        def __signature__(self):
            raise ValueError('Testing Python 3.4')
    f = RaisesValueError()
    assert has_varargs(f) is None