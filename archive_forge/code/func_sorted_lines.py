imports working.
from __future__ import annotations
import glob
import hashlib
import os.path
from typing import Callable, Iterable
from coverage.exceptions import CoverageException, NoDataError
from coverage.files import PathAliases
from coverage.misc import Hasher, file_be_gone, human_sorted, plural
from coverage.sqldata import CoverageData
def sorted_lines(data: CoverageData, filename: str) -> list[int]:
    """Get the sorted lines for a file, for tests."""
    lines = data.lines(filename)
    return sorted(lines or [])