from __future__ import annotations
import hashlib
import ntpath
import os
import os.path
import posixpath
import re
import sys
from typing import Callable, Iterable
from coverage import env
from coverage.exceptions import ConfigError
from coverage.misc import human_sorted, isolate_module, join_regex
def set_relative_directory() -> None:
    """Set the directory that `relative_filename` will be relative to."""
    global RELATIVE_DIR, CANONICAL_FILENAME_CACHE
    abs_curdir = abs_file(os.curdir)
    if not abs_curdir.endswith(os.sep):
        abs_curdir = abs_curdir + os.sep
    RELATIVE_DIR = os.path.normcase(abs_curdir)
    CANONICAL_FILENAME_CACHE = {}