import functools
import inspect
import itertools
import operator
from importlib import import_module
from .functoolz import (is_partial_args, is_arity, has_varargs,
import builtins
def check_partial(sig, args, kwargs):
    """ Like ``is_partial_args`` for the given signature spec"""
    num_pos_only, func, keyword_exclude, sigspec = sig
    if len(args) < num_pos_only:
        pad = (None,) * (num_pos_only - len(args))
        args = args + pad
    if keyword_exclude:
        kwargs = dict(kwargs)
        for item in keyword_exclude:
            kwargs.pop(item, None)
    return is_partial_args(func, args, kwargs, sigspec)