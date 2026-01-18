import atexit
import contextlib
from enum import Enum
from errno import EBADF
from errno import ELOOP
from errno import ENOENT
from errno import ENOTDIR
import fnmatch
from functools import partial
import importlib.util
import itertools
import os
from os.path import expanduser
from os.path import expandvars
from os.path import isabs
from os.path import sep
from pathlib import Path
from pathlib import PurePath
from posixpath import sep as posix_sep
import shutil
import sys
import types
from types import ModuleType
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
import uuid
import warnings
from _pytest.compat import assert_never
from _pytest.outcomes import skip
from _pytest.warning_types import PytestWarning
def insert_missing_modules(modules: Dict[str, ModuleType], module_name: str) -> None:
    """
    Used by ``import_path`` to create intermediate modules when using mode=importlib.

    When we want to import a module as "src.tests.test_foo" for example, we need
    to create empty modules "src" and "src.tests" after inserting "src.tests.test_foo",
    otherwise "src.tests.test_foo" is not importable by ``__import__``.
    """
    module_parts = module_name.split('.')
    child_module: Union[ModuleType, None] = None
    module: Union[ModuleType, None] = None
    child_name: str = ''
    while module_name:
        if module_name not in modules:
            try:
                if not sys.meta_path:
                    raise ModuleNotFoundError
                module = importlib.import_module(module_name)
            except ModuleNotFoundError:
                module = ModuleType(module_name, doc="Empty module created by pytest's importmode=importlib.")
        else:
            module = modules[module_name]
        if child_module:
            if not hasattr(module, child_name):
                setattr(module, child_name, child_module)
                modules[module_name] = module
        child_module, child_name = (module, module_name.rpartition('.')[-1])
        module_parts.pop(-1)
        module_name = '.'.join(module_parts)