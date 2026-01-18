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
def check_return_type(func_name: str, retval: T, annotation: Any, memo: TypeCheckMemo) -> T:
    if _suppression.type_checks_suppressed:
        return retval
    if annotation is NoReturn or annotation is Never:
        exc = TypeCheckError(f'{func_name}() was declared never to return but it did')
        if memo.config.typecheck_fail_callback:
            memo.config.typecheck_fail_callback(exc, memo)
        else:
            raise exc
    try:
        check_type_internal(retval, annotation, memo)
    except TypeCheckError as exc:
        if retval is NotImplemented and annotation is bool:
            func_name = func_name.rsplit('.', 1)[-1]
            if func_name in BINARY_MAGIC_METHODS:
                return retval
        qualname = qualified_name(retval, add_class_prefix=True)
        exc.append_path_element(f'the return value ({qualname})')
        if memo.config.typecheck_fail_callback:
            memo.config.typecheck_fail_callback(exc, memo)
        else:
            raise
    return retval