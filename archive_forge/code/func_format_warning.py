from __future__ import annotations
import errno
import os
import site
import stat
import sys
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional
import platformdirs
from .utils import deprecation
def format_warning(msg: str, *args: Any, **kwargs: Any) -> str:
    return str(msg) + '\n'