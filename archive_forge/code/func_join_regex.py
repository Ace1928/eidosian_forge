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
def join_regex(regexes: Iterable[str]) -> str:
    """Combine a series of regex strings into one that matches any of them."""
    regexes = list(regexes)
    if len(regexes) == 1:
        return regexes[0]
    else:
        return '|'.join((f'(?:{r})' for r in regexes))