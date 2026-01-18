import ctypes
import sys
from typing import Any
import time
from ctypes import Structure, byref, wintypes
from typing import IO, NamedTuple, Type, cast
from pip._vendor.rich.color import ColorSystem
from pip._vendor.rich.style import Style
def erase_start_of_line(self) -> None:
    """Erase all content from the cursor position to the start of that line"""
    row, col = self.cursor_position
    start = WindowsCoordinates(row, 0)
    FillConsoleOutputCharacter(self._handle, ' ', length=col, start=start)
    FillConsoleOutputAttribute(self._handle, self._default_attrs, length=col, start=start)