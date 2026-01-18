import functools
import inspect
import itertools
import operator
from importlib import import_module
from .functoolz import (is_partial_args, is_arity, has_varargs,
import builtins
def get_exclude_keywords(num_pos_only, sigspec):
    """ Return the names of position-only arguments if func has **kwargs"""
    if num_pos_only == 0:
        return ()
    has_kwargs = any((x.kind == x.VAR_KEYWORD for x in sigspec.parameters.values()))
    if not has_kwargs:
        return ()
    pos_args = list(sigspec.parameters.values())[:num_pos_only]
    return tuple((x.name for x in pos_args))