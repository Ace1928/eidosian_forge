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
def stdout_link(text: str, url: str) -> str:
    """Format text+url as a clickable link for stdout.

    If attached to a terminal, use escape sequences. Otherwise, just return
    the text.
    """
    if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
        return f'\x1b]8;;{url}\x07{text}\x1b]8;;\x07'
    else:
        return text