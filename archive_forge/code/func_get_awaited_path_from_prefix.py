from __future__ import unicode_literals
import os.path as op
from send2trash.compat import text_type
from send2trash.util import preprocess_paths
from ctypes import (
from ctypes.wintypes import HWND, UINT, LPCWSTR, BOOL
def get_awaited_path_from_prefix(prefix, path):
    """Guess the correct path to pass to the SHFileOperationW() call.
    The long-path prefix must be removed, so we should take care of
    different long-path prefixes.
    """
    if prefix == '\\\\?\\UNC':
        return '\\' + path[len(prefix):]
    return path[len(prefix):]