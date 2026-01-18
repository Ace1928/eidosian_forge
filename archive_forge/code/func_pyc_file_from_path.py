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
def pyc_file_from_path(path: str) -> Optional[str]:
    """Given a python source path, locate the .pyc."""
    candidate = importlib.util.cache_from_source(path)
    if os.path.exists(candidate):
        return candidate
    filepath, ext = os.path.splitext(path)
    for ext in importlib.machinery.BYTECODE_SUFFIXES:
        if os.path.exists(filepath + ext):
            return filepath + ext
    else:
        return None