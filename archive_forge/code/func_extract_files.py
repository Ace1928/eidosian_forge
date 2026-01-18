from __future__ import annotations
import os
import re
import inspect
import functools
from typing import (
from pathlib import Path
from typing_extensions import TypeGuard
import sniffio
from .._types import Headers, NotGiven, FileTypes, NotGivenOr, HeadersLike
from .._compat import parse_date as parse_date, parse_datetime as parse_datetime
def extract_files(query: Mapping[str, object], *, paths: Sequence[Sequence[str]]) -> list[tuple[str, FileTypes]]:
    """Recursively extract files from the given dictionary based on specified paths.

    A path may look like this ['foo', 'files', '<array>', 'data'].

    Note: this mutates the given dictionary.
    """
    files: list[tuple[str, FileTypes]] = []
    for path in paths:
        files.extend(_extract_items(query, path, index=0, flattened_key=None))
    return files