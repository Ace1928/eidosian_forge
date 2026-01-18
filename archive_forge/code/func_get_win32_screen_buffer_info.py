from __future__ import unicode_literals
from ctypes import windll, byref, ArgumentError, c_char, c_long, c_ulong, c_uint, pointer
from ctypes.wintypes import DWORD
from prompt_toolkit.renderer import Output
from prompt_toolkit.styles import ANSI_COLOR_NAMES
from prompt_toolkit.win32_types import CONSOLE_SCREEN_BUFFER_INFO, STD_OUTPUT_HANDLE, STD_INPUT_HANDLE, COORD, SMALL_RECT
import os
import six
def get_win32_screen_buffer_info(self):
    """
        Return Screen buffer info.
        """
    self.flush()
    sbinfo = CONSOLE_SCREEN_BUFFER_INFO()
    success = windll.kernel32.GetConsoleScreenBufferInfo(self.hconsole, byref(sbinfo))
    if success:
        return sbinfo
    else:
        raise NoConsoleScreenBufferError