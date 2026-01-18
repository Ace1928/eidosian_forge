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
def check_multi_variable_assignment(value: Any, targets: list[dict[str, Any]], memo: TypeCheckMemo) -> Any:
    if max((len(target) for target in targets)) == 1:
        iterated_values = [value]
    else:
        iterated_values = list(value)
    if not _suppression.type_checks_suppressed:
        for expected_types in targets:
            value_index = 0
            for ann_index, (varname, expected_type) in enumerate(expected_types.items()):
                if varname.startswith('*'):
                    varname = varname[1:]
                    keys_left = len(expected_types) - 1 - ann_index
                    next_value_index = len(iterated_values) - keys_left
                    obj: object = iterated_values[value_index:next_value_index]
                    value_index = next_value_index
                else:
                    obj = iterated_values[value_index]
                    value_index += 1
                try:
                    check_type_internal(obj, expected_type, memo)
                except TypeCheckError as exc:
                    qualname = qualified_name(obj, add_class_prefix=True)
                    exc.append_path_element(f'value assigned to {varname} ({qualname})')
                    if memo.config.typecheck_fail_callback:
                        memo.config.typecheck_fail_callback(exc, memo)
                    else:
                        raise
    return iterated_values[0] if len(iterated_values) == 1 else iterated_values