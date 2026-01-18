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
def check_send_type(func_name: str, sendval: T, annotation: Any, memo: TypeCheckMemo) -> T:
    if _suppression.type_checks_suppressed:
        return sendval
    if annotation is NoReturn or annotation is Never:
        exc = TypeCheckError(f'{func_name}() was declared never to be sent a value to but it was')
        if memo.config.typecheck_fail_callback:
            memo.config.typecheck_fail_callback(exc, memo)
        else:
            raise exc
    try:
        check_type_internal(sendval, annotation, memo)
    except TypeCheckError as exc:
        qualname = qualified_name(sendval, add_class_prefix=True)
        exc.append_path_element(f'the value sent to generator ({qualname})')
        if memo.config.typecheck_fail_callback:
            memo.config.typecheck_fail_callback(exc, memo)
        else:
            raise
    return sendval