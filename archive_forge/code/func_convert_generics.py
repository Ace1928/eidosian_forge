import sys
import typing
from collections.abc import Callable
from os import PathLike
from typing import (  # type: ignore
from typing_extensions import (
def convert_generics(tp: Type[Any]) -> Type[Any]:
    """
        Recursively searches for `str` type hints and replaces them with ForwardRef.

        Examples::
            convert_generics(list['Hero']) == list[ForwardRef('Hero')]
            convert_generics(dict['Hero', 'Team']) == dict[ForwardRef('Hero'), ForwardRef('Team')]
            convert_generics(typing.Dict['Hero', 'Team']) == typing.Dict[ForwardRef('Hero'), ForwardRef('Team')]
            convert_generics(list[str | 'Hero'] | int) == list[str | ForwardRef('Hero')] | int
        """
    origin = get_origin(tp)
    if not origin or not hasattr(tp, '__args__'):
        return tp
    args = get_args(tp)
    if origin is Annotated:
        return _AnnotatedAlias(convert_generics(args[0]), args[1:])
    converted = tuple((ForwardRef(arg) if isinstance(arg, str) and isinstance(tp, TypingGenericAlias) else convert_generics(arg) for arg in args))
    if converted == args:
        return tp
    elif isinstance(tp, TypingGenericAlias):
        return TypingGenericAlias(origin, converted)
    elif isinstance(tp, TypesUnionType):
        return _UnionGenericAlias(origin, converted)
    else:
        try:
            setattr(tp, '__args__', converted)
        except AttributeError:
            pass
        return tp