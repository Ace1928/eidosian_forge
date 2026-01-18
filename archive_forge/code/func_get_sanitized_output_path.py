import ctypes
import hashlib
import os
import pathlib
import platform
import sys
import time as _time
import zlib
from datetime import datetime, timedelta, timezone, tzinfo
from typing import BinaryIO, List, Optional, Union
import py7zr.win32compat
from py7zr import Bad7zFile
from py7zr.win32compat import is_windows_native_python, is_windows_unc_path
def get_sanitized_output_path(fname: str, path: Optional[pathlib.Path]) -> pathlib.Path:
    """
    check f.filename has invalid directory traversals
    When condition is not satisfied, raise Bad7zFile
    """
    if fname.startswith('/'):
        fname = fname.lstrip('/')
    if path is None:
        target_path = canonical_path(pathlib.Path.cwd().joinpath(fname))
        if is_relative_to(target_path, pathlib.Path.cwd()):
            return pathlib.Path(remove_relative_path_marker(fname))
    else:
        outfile = canonical_path(path.joinpath(remove_relative_path_marker(fname)))
        if is_relative_to(outfile, path):
            return pathlib.Path(outfile)
    raise Bad7zFile(f'Specified path is bad: {fname}')