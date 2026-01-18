from __future__ import unicode_literals
from ctypes import windll, byref, ArgumentError, c_char, c_long, c_ulong, c_uint, pointer
from ctypes.wintypes import DWORD
from prompt_toolkit.renderer import Output
from prompt_toolkit.styles import ANSI_COLOR_NAMES
from prompt_toolkit.win32_types import CONSOLE_SCREEN_BUFFER_INFO, STD_OUTPUT_HANDLE, STD_INPUT_HANDLE, COORD, SMALL_RECT
import os
import six
def scroll_buffer_to_prompt(self):
    """
        To be called before drawing the prompt. This should scroll the console
        to left, with the cursor at the bottom (if possible).
        """
    info = self.get_win32_screen_buffer_info()
    sr = info.srWindow
    cursor_pos = info.dwCursorPosition
    result = SMALL_RECT()
    result.Left = 0
    result.Right = sr.Right - sr.Left
    win_height = sr.Bottom - sr.Top
    if 0 < sr.Bottom - cursor_pos.Y < win_height - 1:
        result.Bottom = sr.Bottom
    else:
        result.Bottom = max(win_height, cursor_pos.Y)
    result.Top = result.Bottom - win_height
    self._winapi(windll.kernel32.SetConsoleWindowInfo, self.hconsole, True, byref(result))