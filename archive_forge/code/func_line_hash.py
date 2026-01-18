from __future__ import annotations
import base64
import hashlib
import sys
from typing import IO, Iterable, TYPE_CHECKING
from coverage.plugin import FileReporter
from coverage.report_core import get_analysis_to_report
from coverage.results import Analysis, Numbers
from coverage.types import TMorf
def line_hash(line: str) -> str:
    """Produce a hash of a source line for use in the LCOV file."""
    hashed = hashlib.md5(line.encode('utf-8')).digest()
    return base64.b64encode(hashed).decode('ascii').rstrip('=')