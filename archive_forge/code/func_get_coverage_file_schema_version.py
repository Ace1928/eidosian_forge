from __future__ import annotations
import dataclasses
import os
import sqlite3
import tempfile
import typing as t
from .config import (
from .io import (
from .util import (
from .data import (
from .util_common import (
from .host_configs import (
from .constants import (
from .thread import (
def get_coverage_file_schema_version(path: str) -> int:
    """
    Return the schema version from the specified coverage file.
    SQLite based files report schema version 1 or later.
    JSON based files are reported as schema version 0.
    An exception is raised if the file is not recognized or the schema version cannot be determined.
    """
    with open_binary_file(path) as file_obj:
        header = file_obj.read(16)
    if header.startswith(b'!coverage.py: '):
        return 0
    if header.startswith(b'SQLite'):
        return get_sqlite_schema_version(path)
    raise CoverageError(path, f'Unknown header: {header!r}')