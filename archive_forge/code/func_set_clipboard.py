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
def set_clipboard(clipboard):
    """
    Explicitly sets the clipboard mechanism. The "clipboard mechanism" is how
    the copy() and paste() functions interact with the operating system to
    implement the copy/paste feature. The clipboard parameter must be one of:
        - pbcopy
        - pyobjc (default on macOS)
        - qt
        - xclip
        - xsel
        - klipper
        - windows (default on Windows)
        - no (this is what is set when no clipboard mechanism can be found)
    """
    global copy, paste
    clipboard_types = {'pbcopy': init_osx_pbcopy_clipboard, 'pyobjc': init_osx_pyobjc_clipboard, 'qt': init_qt_clipboard, 'xclip': init_xclip_clipboard, 'xsel': init_xsel_clipboard, 'wl-clipboard': init_wl_clipboard, 'klipper': init_klipper_clipboard, 'windows': init_windows_clipboard, 'no': init_no_clipboard}
    if clipboard not in clipboard_types:
        allowed_clipboard_types = [repr(_) for _ in clipboard_types]
        raise ValueError(f'Argument must be one of {', '.join(allowed_clipboard_types)}')
    copy, paste = clipboard_types[clipboard]()