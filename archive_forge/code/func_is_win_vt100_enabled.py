from __future__ import annotations
import sys
from ctypes import byref, windll
from ctypes.wintypes import DWORD, HANDLE
from typing import Any, TextIO
from prompt_toolkit.data_structures import Size
from prompt_toolkit.win32_types import STD_OUTPUT_HANDLE
from .base import Output
from .color_depth import ColorDepth
from .vt100 import Vt100_Output
from .win32 import Win32Output
def is_win_vt100_enabled() -> bool:
    """
    Returns True when we're running Windows and VT100 escape sequences are
    supported.
    """
    if sys.platform != 'win32':
        return False
    hconsole = HANDLE(windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE))
    original_mode = DWORD(0)
    windll.kernel32.GetConsoleMode(hconsole, byref(original_mode))
    try:
        result: int = windll.kernel32.SetConsoleMode(hconsole, DWORD(ENABLE_PROCESSED_INPUT | ENABLE_VIRTUAL_TERMINAL_PROCESSING))
        return result == 1
    finally:
        windll.kernel32.SetConsoleMode(hconsole, original_mode)