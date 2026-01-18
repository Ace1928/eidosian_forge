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
def deprecate_attributes(module_name: str, deprecated_attributes: Dict[str, Tuple[str, str]]):
    """Replace module with a wrapper that gives warnings for deprecated attributes.

    Args:
        module_name: Absolute name of the module that deprecates attributes.
        deprecated_attributes: A dictionary from attribute name to a tuple of
            strings, where the first string gives the version that the attribute
            will be removed in, and the second string describes what the user
            should do instead of accessing this deprecated attribute.

    Returns:
        Wrapped module with deprecated attributes. Use of these attributes
        will cause a warning for these deprecated attributes.
    """
    for deadline, _ in deprecated_attributes.values():
        _validate_deadline(deadline)
    module = sys.modules[module_name]

    class Wrapped(ModuleType):
        __dict__ = module.__dict__
        __spec__ = _make_proxy_spec_property(module)

        def __getattr__(self, name):
            if name in deprecated_attributes:
                deadline, fix = deprecated_attributes[name]
                _warn_or_error(f'{name} was used but is deprecated.\nIt will be removed in cirq {deadline}.\n{fix}\n')
            return getattr(module, name)
    wrapped_module = Wrapped(module_name, module.__doc__)
    if '.' in module_name:
        parent_name, module_tail = module_name.rsplit('.', 1)
        setattr(sys.modules[parent_name], module_tail, wrapped_module)
    sys.modules[module_name] = wrapped_module
    return wrapped_module