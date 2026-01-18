import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
def hash_args_kwargs(fn_name, *args, **kwargs):
    hargs = tuple(map(maybe_id, args))
    if kwargs:
        hkwargs = tuple(sorted(((k, maybe_id(v)) for k, v in kwargs.items())))
    else:
        hkwargs = None
    return f'{fn_name}-{hash((hargs, hkwargs))}'