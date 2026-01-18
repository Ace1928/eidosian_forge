from __future__ import with_statement
import ctypes.wintypes
from functools import reduce
def get_directory_handle(path):
    """Returns a Windows handle to the specified directory path."""
    return CreateFileW(path, FILE_LIST_DIRECTORY, WATCHDOG_FILE_SHARE_FLAGS, None, OPEN_EXISTING, WATCHDOG_FILE_FLAGS, None)