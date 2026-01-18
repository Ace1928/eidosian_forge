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
def coerce_resource_to_filename(fname: str) -> str:
    """Interpret a filename as either a filesystem location or as a package
    resource.

    Names that are non absolute paths and contain a colon
    are interpreted as resources and coerced to a file location.

    """
    if not os.path.isabs(fname) and ':' in fname:
        tokens = fname.split(':')
        file_manager = ExitStack()
        atexit.register(file_manager.close)
        ref = compat.importlib_resources.files(tokens[0])
        for tok in tokens[1:]:
            ref = ref / tok
        fname = file_manager.enter_context(compat.importlib_resources.as_file(ref))
    return fname