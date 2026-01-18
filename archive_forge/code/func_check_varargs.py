import functools
import inspect
import itertools
import operator
from importlib import import_module
from .functoolz import (is_partial_args, is_arity, has_varargs,
import builtins
def check_varargs(sig):
    num_pos_only, func, keyword_exclude, sigspec = sig
    return has_varargs(func, sigspec)