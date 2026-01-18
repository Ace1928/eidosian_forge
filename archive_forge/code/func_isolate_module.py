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
def isolate_module(mod: ModuleType) -> ModuleType:
    """Copy a module so that we are isolated from aggressive mocking.

    If a test suite mocks os.path.exists (for example), and then we need to use
    it during the test, everything will get tangled up if we use their mock.
    Making a copy of the module when we import it will isolate coverage.py from
    those complications.
    """
    if mod not in ISOLATED_MODULES:
        new_mod = types.ModuleType(mod.__name__)
        ISOLATED_MODULES[mod] = new_mod
        for name in dir(mod):
            value = getattr(mod, name)
            if isinstance(value, types.ModuleType):
                value = isolate_module(value)
            setattr(new_mod, name, value)
    return ISOLATED_MODULES[mod]