from __future__ import annotations
import importlib.machinery
import importlib.util
import inspect
import marshal
import os
import struct
import sys
from importlib.machinery import ModuleSpec
from types import CodeType, ModuleType
from typing import Any
from coverage import env
from coverage.exceptions import CoverageException, _ExceptionDuringRun, NoCode, NoSource
from coverage.files import canonical_filename, python_reported_file
from coverage.misc import isolate_module
from coverage.python import get_python_source
def make_code_from_py(filename: str) -> CodeType:
    """Get source from `filename` and make a code object of it."""
    try:
        source = get_python_source(filename)
    except (OSError, NoSource) as exc:
        raise NoSource(f"No file to run: '{filename}'") from exc
    return compile(source, filename, 'exec', dont_inherit=True)