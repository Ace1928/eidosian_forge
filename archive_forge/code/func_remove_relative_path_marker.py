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
def remove_relative_path_marker(path: str) -> str:
    """
    Removes './' from the beginning of a path-like string
    """
    processed_path = path
    if path.startswith(RELATIVE_PATH_MARKER):
        processed_path = path[len(RELATIVE_PATH_MARKER):]
    return processed_path