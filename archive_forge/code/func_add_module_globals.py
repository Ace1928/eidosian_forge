from __future__ import annotations as _annotations
import dataclasses
import sys
import types
import typing
from collections.abc import Callable
from functools import partial
from types import GetSetDescriptorType
from typing import TYPE_CHECKING, Any, Final
from typing_extensions import Annotated, Literal, TypeAliasType, TypeGuard, get_args, get_origin
def add_module_globals(obj: Any, globalns: dict[str, Any] | None=None) -> dict[str, Any]:
    module_name = getattr(obj, '__module__', None)
    if module_name:
        try:
            module_globalns = sys.modules[module_name].__dict__
        except KeyError:
            pass
        else:
            if globalns:
                return {**module_globalns, **globalns}
            else:
                return module_globalns.copy()
    return globalns or {}