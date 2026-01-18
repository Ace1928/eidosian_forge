from __future__ import annotations
import sys
import warnings
from typing import Any, Callable, NoReturn, TypeVar, Union, overload
from . import _suppression
from ._checkers import BINARY_MAGIC_METHODS, check_type_internal
from ._config import (
from ._exceptions import TypeCheckError, TypeCheckWarning
from ._memo import TypeCheckMemo
from ._utils import get_stacklevel, qualified_name
def check_argument_types(func_name: str, arguments: dict[str, tuple[Any, Any]], memo: TypeCheckMemo) -> Literal[True]:
    if _suppression.type_checks_suppressed:
        return True
    for argname, (value, annotation) in arguments.items():
        if annotation is NoReturn or annotation is Never:
            exc = TypeCheckError(f'{func_name}() was declared never to be called but it was')
            if memo.config.typecheck_fail_callback:
                memo.config.typecheck_fail_callback(exc, memo)
            else:
                raise exc
        try:
            check_type_internal(value, annotation, memo)
        except TypeCheckError as exc:
            qualname = qualified_name(value, add_class_prefix=True)
            exc.append_path_element(f'argument "{argname}" ({qualname})')
            if memo.config.typecheck_fail_callback:
                memo.config.typecheck_fail_callback(exc, memo)
            else:
                raise
    return True