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
def run_python_file(args: list[str]) -> None:
    """Run a Python file as if it were the main program on the command line.

    `args` is the argument array to present as sys.argv, including the first
    element naming the file being executed.  `package` is the name of the
    enclosing package, if any.

    This is a helper for tests, to encapsulate how to use PyRunner.

    """
    runner = PyRunner(args, as_module=False)
    runner.prepare()
    runner.run()