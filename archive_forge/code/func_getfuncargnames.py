from __future__ import annotations
import dataclasses
import enum
import functools
import inspect
from inspect import Parameter
from inspect import signature
import os
from pathlib import Path
import sys
from typing import Any
from typing import Callable
from typing import Final
from typing import NoReturn
import py
def getfuncargnames(function: Callable[..., object], *, name: str='', is_method: bool=False, cls: type | None=None) -> tuple[str, ...]:
    """Return the names of a function's mandatory arguments.

    Should return the names of all function arguments that:
    * Aren't bound to an instance or type as in instance or class methods.
    * Don't have default values.
    * Aren't bound with functools.partial.
    * Aren't replaced with mocks.

    The is_method and cls arguments indicate that the function should
    be treated as a bound method even though it's not unless, only in
    the case of cls, the function is a static method.

    The name parameter should be the original name in which the function was collected.
    """
    try:
        parameters = signature(function).parameters
    except (ValueError, TypeError) as e:
        from _pytest.outcomes import fail
        fail(f'Could not determine arguments of {function!r}: {e}', pytrace=False)
    arg_names = tuple((p.name for p in parameters.values() if (p.kind is Parameter.POSITIONAL_OR_KEYWORD or p.kind is Parameter.KEYWORD_ONLY) and p.default is Parameter.empty))
    if not name:
        name = function.__name__
    if is_method or (cls and (not isinstance(inspect.getattr_static(cls, name, default=None), staticmethod))):
        arg_names = arg_names[1:]
    if hasattr(function, '__wrapped__'):
        arg_names = arg_names[num_mock_patch_args(function):]
    return arg_names