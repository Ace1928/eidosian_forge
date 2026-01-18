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
def should_be_python(self) -> bool:
    """Does it seem like this file should contain Python?

        This is used to decide if a file reported as part of the execution of
        a program was really likely to have contained Python in the first
        place.

        """
    _, ext = os.path.splitext(self.filename)
    if ext.startswith('.py'):
        return True
    if not ext:
        return True
    return False