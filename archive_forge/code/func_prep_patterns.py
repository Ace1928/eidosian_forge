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
def prep_patterns(patterns: Iterable[str]) -> list[str]:
    """Prepare the file patterns for use in a `GlobMatcher`.

    If a pattern starts with a wildcard, it is used as a pattern
    as-is.  If it does not start with a wildcard, then it is made
    absolute with the current directory.

    If `patterns` is None, an empty list is returned.

    """
    prepped = []
    for p in patterns or []:
        prepped.append(p)
        if not p.startswith(('*', '?')):
            prepped.append(abs_file(p))
    return prepped