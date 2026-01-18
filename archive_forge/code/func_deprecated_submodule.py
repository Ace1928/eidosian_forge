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
def deprecated_submodule(*, new_module_name: str, old_parent: str, old_child: str, deadline: str, create_attribute: bool):
    """Creates a deprecated module reference recursively for a module.

    For `new_module_name` (e.g. cirq_google) creates an alias (e.g cirq.google) in Python's module
    cache. It also recursively checks for the already imported submodules (e.g. cirq_google.api) and
    creates the alias for them too (e.g. cirq.google.api). With this method it is possible to create
    an alias that really looks like a module, e.g you can do things like
    `from cirq.google import api` - which would be otherwise impossible.

    Note that this method will execute `new_module_name` in order to ensure that it is in the module
    cache.

    Args:
        new_module_name: Absolute module name for the new module.
        old_parent: The current module that had the original submodule.
        old_child: The submodule that is being relocated.
        deadline: The version of Cirq where the module will be removed.
        create_attribute: If True, the submodule will be added as a deprecated attribute to the
            old_parent module.

    Returns:
        None
    """
    _validate_deadline(deadline)
    old_module_name = f'{old_parent}.{old_child}'
    broken_module_exception = None
    if create_attribute:
        try:
            new_module = importlib.import_module(new_module_name)
            _setup_deprecated_submodule_attribute(new_module_name, old_parent, old_child, deadline, new_module)
        except ImportError as ex:
            msg = f"{new_module_name} cannot be imported. The typical reasons are that\n 1.) {new_module_name} is not installed, or\n 2.) when developing Cirq, you don't have your PYTHONPATH setup. In this case run `source dev_tools/pypath`.\n\n You can check the detailed exception above for more details or run `import {new_module_name} to reproduce the issue."
            broken_module_exception = DeprecatedModuleImportError(msg)
            broken_module_exception.__cause__ = ex
            _setup_deprecated_submodule_attribute(new_module_name, old_parent, old_child, deadline, _BrokenModule(new_module_name, broken_module_exception))
    finder = DeprecatedModuleFinder(new_module_name, old_module_name, deadline, broken_module_exception)
    sys.meta_path.append(finder)