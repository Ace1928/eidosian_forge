import functools
import inspect
import itertools
import operator
import toolz
from toolz.functoolz import (curry, is_valid_args, is_partial_args, is_arity,
from toolz._signatures import builtins
import toolz._signatures as _sigs
from toolz.utils import raises
def check_curry(func, args, kwargs, incomplete=True):
    try:
        curry(func)(*args, **kwargs)
        curry(func, *args)(**kwargs)
        curry(func, **kwargs)(*args)
        curry(func, *args, **kwargs)()
        if not isinstance(func, type(lambda: None)):
            return None
        return incomplete
    except ValueError:
        return True
    except TypeError:
        return False