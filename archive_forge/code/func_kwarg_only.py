from collections.abc import Sequence
import functools
import sys
from typing import Any, NamedTuple, Optional, Protocol, TypeVar
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
def kwarg_only(f: Any) -> Any:
    """A wrapper that throws away all non-kwarg arguments."""
    f_argspec = tf_inspect.getfullargspec(f)

    def wrapper(*args, **kwargs):
        if args:
            raise TypeError('{f} only takes keyword args (possible keys: {kwargs}). Please pass these args as kwargs instead.'.format(f=f.__name__, kwargs=f_argspec.args))
        return f(**kwargs)
    return tf_decorator.make_decorator(f, wrapper, decorator_argspec=f_argspec)