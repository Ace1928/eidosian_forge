import typing
import types
import inspect
import functools
from . import _uarray
import copyreg
import pickle
import contextlib
from ._uarray import (  # type: ignore
def get_defaults(f):
    sig = inspect.signature(f)
    kw_defaults = {}
    arg_defaults = []
    opts = set()
    for k, v in sig.parameters.items():
        if v.default is not inspect.Parameter.empty:
            kw_defaults[k] = v.default
        if v.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            arg_defaults.append(v.default)
        opts.add(k)
    return (kw_defaults, tuple(arg_defaults), opts)