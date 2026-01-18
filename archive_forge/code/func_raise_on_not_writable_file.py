from __future__ import annotations
import os
import stat
import sys
from errno import EACCES, EISDIR
from pathlib import Path
def raise_on_not_writable_file(filename: str) -> None:
    """
    Raise an exception if attempting to open the file for writing would fail.
    This is done so files that will never be writable can be separated from
    files that are writable but currently locked
    :param filename: file to check
    :raises OSError: as if the file was opened for writing.
    """
    try:
        file_stat = os.stat(filename)
    except OSError:
        return
    if file_stat.st_mtime != 0:
        if not file_stat.st_mode & stat.S_IWUSR:
            raise PermissionError(EACCES, 'Permission denied', filename)
        if stat.S_ISDIR(file_stat.st_mode):
            if sys.platform == 'win32':
                raise PermissionError(EACCES, 'Permission denied', filename)
            else:
                raise IsADirectoryError(EISDIR, 'Is a directory', filename)