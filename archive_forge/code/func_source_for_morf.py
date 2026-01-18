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
def source_for_morf(morf: TMorf) -> str:
    """Get the source filename for the module-or-file `morf`."""
    if hasattr(morf, '__file__') and morf.__file__:
        filename = morf.__file__
    elif isinstance(morf, types.ModuleType):
        raise CoverageException(f'Module {morf} has no file')
    else:
        filename = morf
    filename = source_for_file(filename)
    return filename