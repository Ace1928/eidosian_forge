import collections.abc
import gc
import inspect
import re
import sys
import weakref
from functools import partial, wraps
from itertools import chain
from typing import (
from scrapy.utils.asyncgen import as_async_generator
def get_spec(func: Callable) -> Tuple[List[str], Dict[str, Any]]:
    """Returns (args, kwargs) tuple for a function
    >>> import re
    >>> get_spec(re.match)
    (['pattern', 'string'], {'flags': 0})

    >>> class Test:
    ...     def __call__(self, val):
    ...         pass
    ...     def method(self, val, flags=0):
    ...         pass

    >>> get_spec(Test)
    (['self', 'val'], {})

    >>> get_spec(Test.method)
    (['self', 'val'], {'flags': 0})

    >>> get_spec(Test().method)
    (['self', 'val'], {'flags': 0})
    """
    if inspect.isfunction(func) or inspect.ismethod(func):
        spec = inspect.getfullargspec(func)
    elif hasattr(func, '__call__'):
        spec = inspect.getfullargspec(func.__call__)
    else:
        raise TypeError(f'{type(func)} is not callable')
    defaults: Tuple[Any, ...] = spec.defaults or ()
    firstdefault = len(spec.args) - len(defaults)
    args = spec.args[:firstdefault]
    kwargs = dict(zip(spec.args[firstdefault:], defaults))
    return (args, kwargs)