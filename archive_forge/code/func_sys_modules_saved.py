from __future__ import annotations
import contextlib
import datetime
import errno
import hashlib
import importlib
import importlib.util
import inspect
import locale
import os
import os.path
import re
import sys
import types
from types import ModuleType
from typing import (
from coverage import env
from coverage.exceptions import CoverageException
from coverage.types import TArc
from coverage.exceptions import *   # pylint: disable=wildcard-import
@contextlib.contextmanager
def sys_modules_saved() -> Iterator[None]:
    """A context manager to remove any modules imported during a block."""
    saver = SysModuleSaver()
    try:
        yield
    finally:
        saver.restore()