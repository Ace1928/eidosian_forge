from __future__ import annotations
from .. import mesonlib, mlog
from .disabler import Disabler
from .exceptions import InterpreterException, InvalidArguments
from ._unholder import _unholder
from dataclasses import dataclass
from functools import wraps
import abc
import itertools
import copy
import typing as T
def stringArgs(f: TV_func) -> TV_func:

    @wraps(f)
    def wrapped(*wrapped_args: T.Any, **wrapped_kwargs: T.Any) -> T.Any:
        args = get_callee_args(wrapped_args)[1]
        if not isinstance(args, list):
            mlog.debug('Not a list:', str(args))
            raise InvalidArguments('Argument not a list.')
        if not all((isinstance(s, str) for s in args)):
            mlog.debug('Element not a string:', str(args))
            raise InvalidArguments('Arguments must be strings.')
        return f(*wrapped_args, **wrapped_kwargs)
    return T.cast('TV_func', wrapped)