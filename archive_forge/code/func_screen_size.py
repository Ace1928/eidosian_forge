import ctypes
import sys
from typing import Any
import time
from ctypes import Structure, byref, wintypes
from typing import IO, NamedTuple, Type, cast
from pip._vendor.rich.color import ColorSystem
from pip._vendor.rich.style import Style
@property
def screen_size(self) -> WindowsCoordinates:
    """Returns the current size of the console screen buffer, in character columns and rows

        Returns:
            WindowsCoordinates: The width and height of the screen as WindowsCoordinates.
        """
    screen_size: COORD = GetConsoleScreenBufferInfo(self._handle).dwSize
    return WindowsCoordinates(row=cast(int, screen_size.Y), col=cast(int, screen_size.X))