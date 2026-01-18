from __future__ import unicode_literals
from ctypes import windll, byref, ArgumentError, c_char, c_long, c_ulong, c_uint, pointer
from ctypes.wintypes import DWORD
from prompt_toolkit.renderer import Output
from prompt_toolkit.styles import ANSI_COLOR_NAMES
from prompt_toolkit.win32_types import CONSOLE_SCREEN_BUFFER_INFO, STD_OUTPUT_HANDLE, STD_INPUT_HANDLE, COORD, SMALL_RECT
import os
import six
def reset_attributes(self):
    """ Reset the console foreground/background color. """
    self._winapi(windll.kernel32.SetConsoleTextAttribute, self.hconsole, self.default_attrs)