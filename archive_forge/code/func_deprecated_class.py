import contextlib
import contextvars
import dataclasses
import functools
import importlib
import inspect
import os
import re
import sys
import traceback
import warnings
from types import ModuleType
from typing import Any, Callable, Dict, Iterator, Optional, overload, Set, Tuple, Type, TypeVar
import numpy as np
import pandas as pd
import sympy
import sympy.printing.repr
from cirq._doc import document
def deprecated_class(*, deadline: str, fix: str, name: Optional[str]=None) -> Callable[[Type], Type]:
    """Marks a class as deprecated.

    Args:
        deadline: The version where the function will be deleted. It should be a minor version
            (e.g. "v0.7").
        fix: A complete sentence describing what the user should be using
            instead of this particular function (e.g. "Use cos instead.")
        name: How to refer to the class.
            Defaults to `class.__qualname__`.

    Returns:
        A decorator that decorates classes with a deprecation warning.
    """
    _validate_deadline(deadline)

    def decorator(clazz: Type) -> Type:
        clazz_new = clazz.__new__

        def patched_new(cls, *args, **kwargs):
            qualname = clazz.__qualname__ if name is None else name
            _warn_or_error(f'{qualname} was used but is deprecated.\nIt will be removed in cirq {deadline}.\n{fix}\n')
            return clazz_new(cls)
        setattr(clazz, '__new__', patched_new)
        clazz.__doc__ = f'THIS CLASS IS DEPRECATED.\n\nIT WILL BE REMOVED IN `cirq {deadline}`.\n\n{fix}\n\n{clazz.__doc__ or ''}'
        return clazz
    return decorator