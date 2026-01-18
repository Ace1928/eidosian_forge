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
def get_zip_bytes(filename: str) -> bytes | None:
    """Get data from `filename` if it is a zip file path.

    Returns the bytestring data read from the zip file, or None if no zip file
    could be found or `filename` isn't in it.  The data returned will be
    an empty string if the file is empty.

    """
    zipfile_inner = zip_location(filename)
    if zipfile_inner is not None:
        zipfile, inner = zipfile_inner
        try:
            zi = zipimport.zipimporter(zipfile)
        except zipimport.ZipImportError:
            return None
        try:
            data = zi.get_data(inner)
        except OSError:
            return None
        return data
    return None