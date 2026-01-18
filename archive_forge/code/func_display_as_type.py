from __future__ import annotations as _annotations
import types
import typing
from typing import Any
import typing_extensions
from . import _typing_extra
def display_as_type(obj: Any) -> str:
    """Pretty representation of a type, should be as close as possible to the original type definition string.

    Takes some logic from `typing._type_repr`.
    """
    if isinstance(obj, types.FunctionType):
        return obj.__name__
    elif obj is ...:
        return '...'
    elif isinstance(obj, Representation):
        return repr(obj)
    elif isinstance(obj, typing_extensions.TypeAliasType):
        return str(obj)
    if not isinstance(obj, (_typing_extra.typing_base, _typing_extra.WithArgsTypes, type)):
        obj = obj.__class__
    if _typing_extra.origin_is_union(typing_extensions.get_origin(obj)):
        args = ', '.join(map(display_as_type, typing_extensions.get_args(obj)))
        return f'Union[{args}]'
    elif isinstance(obj, _typing_extra.WithArgsTypes):
        if typing_extensions.get_origin(obj) == typing_extensions.Literal:
            args = ', '.join(map(repr, typing_extensions.get_args(obj)))
        else:
            args = ', '.join(map(display_as_type, typing_extensions.get_args(obj)))
        try:
            return f'{obj.__qualname__}[{args}]'
        except AttributeError:
            return str(obj)
    elif isinstance(obj, type):
        return obj.__qualname__
    else:
        return repr(obj).replace('typing.', '').replace('typing_extensions.', '')