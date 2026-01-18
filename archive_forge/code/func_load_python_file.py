from __future__ import annotations
import atexit
from contextlib import ExitStack
import importlib
import importlib.machinery
import importlib.util
import os
import re
import tempfile
from types import ModuleType
from typing import Any
from typing import Optional
from mako import exceptions
from mako.template import Template
from . import compat
from .exc import CommandError
def load_python_file(dir_: str, filename: str) -> ModuleType:
    """Load a file from the given path as a Python module."""
    module_id = re.sub('\\W', '_', filename)
    path = os.path.join(dir_, filename)
    _, ext = os.path.splitext(filename)
    if ext == '.py':
        if os.path.exists(path):
            module = load_module_py(module_id, path)
        else:
            pyc_path = pyc_file_from_path(path)
            if pyc_path is None:
                raise ImportError("Can't find Python file %s" % path)
            else:
                module = load_module_py(module_id, pyc_path)
    elif ext in ('.pyc', '.pyo'):
        module = load_module_py(module_id, path)
    else:
        assert False
    return module