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
def load_module_py(module_id: str, path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_id, path)
    assert spec
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module