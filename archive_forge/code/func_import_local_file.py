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
def import_local_file(modname: str, modfile: str | None=None) -> ModuleType:
    """Import a local file as a module.

    Opens a file in the current directory named `modname`.py, imports it
    as `modname`, and returns the module object.  `modfile` is the file to
    import if it isn't in the current directory.

    """
    if modfile is None:
        modfile = modname + '.py'
    spec = importlib.util.spec_from_file_location(modname, modfile)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod