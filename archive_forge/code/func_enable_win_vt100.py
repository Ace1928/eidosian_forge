import sys
from enum import (
from typing import (
def enable_win_vt100(handle: HANDLE) -> bool:
    """
            Enables VT100 character sequences in a Windows console
            This only works on Windows 10 and up
            :param handle: the handle on which to enable vt100
            :return: True if vt100 characters are enabled for the handle
            """
    ENABLE_VIRTUAL_TERMINAL_PROCESSING = 4
    cur_mode = DWORD(0)
    readline.rl.console.GetConsoleMode(handle, byref(cur_mode))
    retVal = False
    if cur_mode.value & ENABLE_VIRTUAL_TERMINAL_PROCESSING != 0:
        retVal = True
    elif readline.rl.console.SetConsoleMode(handle, cur_mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING):
        atexit.register(readline.rl.console.SetConsoleMode, handle, cur_mode)
        retVal = True
    return retVal