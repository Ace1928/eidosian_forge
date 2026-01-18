import ctypes
import sys
from typing import Any
import time
from ctypes import Structure, byref, wintypes
from typing import IO, NamedTuple, Type, cast
from pip._vendor.rich.color import ColorSystem
from pip._vendor.rich.style import Style
def move_cursor_backward(self) -> None:
    """Move the cursor backward a single cell. Wrap to the previous line if required."""
    row, col = self.cursor_position
    if col == 0:
        row -= 1
        col = self.screen_size.col - 1
    else:
        col -= 1
    SetConsoleCursorPosition(self._handle, coords=WindowsCoordinates(row=row, col=col))