from __future__ import annotations
import os.path
import types
import zipimport
from typing import Iterable, TYPE_CHECKING
from coverage import env
from coverage.exceptions import CoverageException, NoSource
from coverage.files import canonical_filename, relative_filename, zip_location
from coverage.misc import expensive, isolate_module, join_regex
from coverage.parser import PythonParser
from coverage.phystokens import source_token_lines, source_encoding
from coverage.plugin import FileReporter
from coverage.types import TArc, TLineNo, TMorf, TSourceTokenLines
def read_python_source(filename: str) -> bytes:
    """Read the Python source text from `filename`.

    Returns bytes.

    """
    with open(filename, 'rb') as f:
        source = f.read()
    return source.replace(b'\r\n', b'\n').replace(b'\r', b'\n')