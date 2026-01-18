import contextlib
import ctypes
from ctypes import (
import os
import platform
from shutil import which as _executable_exists
import subprocess
import time
import warnings
from pandas.errors import (
from pandas.util._exceptions import find_stack_level
def copy_windows(text):
    text = _stringifyText(text)
    with window() as hwnd:
        with clipboard(hwnd):
            safeEmptyClipboard()
            if text:
                count = wcslen(text) + 1
                handle = safeGlobalAlloc(GMEM_MOVEABLE, count * sizeof(c_wchar))
                locked_handle = safeGlobalLock(handle)
                ctypes.memmove(c_wchar_p(locked_handle), c_wchar_p(text), count * sizeof(c_wchar))
                safeGlobalUnlock(handle)
                safeSetClipboardData(CF_UNICODETEXT, handle)