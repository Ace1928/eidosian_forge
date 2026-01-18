from __future__ import annotations
from collections.abc import Callable, Sequence
from functools import partial
from inspect import getmro, isclass
from typing import TYPE_CHECKING, Generic, Type, TypeVar, cast, overload
def get_condition_filter(condition: type[_BaseExceptionT] | tuple[type[_BaseExceptionT], ...] | Callable[[_BaseExceptionT_co], bool]) -> Callable[[_BaseExceptionT_co], bool]:
    if isclass(condition) and issubclass(cast(Type[BaseException], condition), BaseException):
        return partial(check_direct_subclass, parents=(condition,))
    elif isinstance(condition, tuple):
        if all((isclass(x) and issubclass(x, BaseException) for x in condition)):
            return partial(check_direct_subclass, parents=condition)
    elif callable(condition):
        return cast('Callable[[BaseException], bool]', condition)
    raise TypeError('expected a function, exception type or tuple of exception types')