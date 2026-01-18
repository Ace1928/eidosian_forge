import collections.abc
import copy
import dataclasses
import inspect
import sys
import types
import warnings
from typing import (
from typing_extensions import Annotated, Self, get_args, get_origin, get_type_hints
from . import _fields, _unsafe_cache
from ._typing import TypeForm
def get_type_hints_with_nicer_errors(obj: Callable[..., Any], include_extras: bool=False) -> Dict[str, Any]:
    try:
        return get_type_hints(obj, include_extras=include_extras)
    except TypeError as e:
        common_message = f'\n-----\n\nError calling typing.get_type_hints() on {obj}! '
        message = e.args[0]
        if message.startswith('unsupported operand type(s) for |') and sys.version_info < (3, 10):
            raise TypeError(e.args[0] + common_message + 'You may be using a Union in the form of `X | Y`; to support Python versions that lower than 3.10, you need to use `typing.Union[X, Y]` instead of `X | Y` and `typing.Optional[X]` instead of `X | None`.', *e.args[1:])
        elif message == "'type' object is not subscriptable" and sys.version_info < (3, 9):
            raise TypeError(e.args[0] + common_message + 'You may be using a standard collection as a generic, like `list[T]` or `tuple[T1, T2]`. To support Python versions lower than 3.9, you need to using `typing.List[T]` and `typing.Tuple[T1, T2]`.', *e.args[1:])
        raise e