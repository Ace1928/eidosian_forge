from __future__ import annotations
import atexit
import contextlib
import functools
import inspect
import itertools
import os
import pprint
import re
import reprlib
import sys
import traceback
import types
import _thread
from typing import (
from coverage.misc import human_sorted_items, isolate_module
from coverage.types import AnyCallable, TWritable
def short_filename(filename: str | None) -> str | None:
    """Shorten a file name. Directories are replaced by prefixes like 'syspath:'"""
    if not _FILENAME_SUBS:
        for pathdir in sys.path:
            _FILENAME_SUBS.append((pathdir, 'syspath:'))
        import coverage
        _FILENAME_SUBS.append((os.path.dirname(coverage.__file__), 'cov:'))
        _FILENAME_SUBS.sort(key=lambda pair: len(pair[0]), reverse=True)
    if filename is not None:
        for pat, sub in _FILENAME_REGEXES:
            filename = re.sub(pat, sub, filename)
        for before, after in _FILENAME_SUBS:
            filename = filename.replace(before, after)
    return filename