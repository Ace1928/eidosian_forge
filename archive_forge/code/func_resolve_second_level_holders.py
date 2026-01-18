from __future__ import annotations
from .. import mesonlib, mparser
from .exceptions import InterpreterException, InvalidArguments
from ..coredata import UserOption
import collections.abc
import typing as T
def resolve_second_level_holders(args: T.List['TYPE_var'], kwargs: 'TYPE_kwargs') -> T.Tuple[T.List['TYPE_var'], 'TYPE_kwargs']:

    def resolver(arg: 'TYPE_var') -> 'TYPE_var':
        if isinstance(arg, list):
            return [resolver(x) for x in arg]
        if isinstance(arg, dict):
            return {k: resolver(v) for k, v in arg.items()}
        if isinstance(arg, mesonlib.SecondLevelHolder):
            return arg.get_default_object()
        return arg
    return ([resolver(x) for x in args], {k: resolver(v) for k, v in kwargs.items()})