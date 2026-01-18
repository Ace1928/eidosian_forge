import ctypes
import sys
from typing import Any
import time
from ctypes import Structure, byref, wintypes
from typing import IO, NamedTuple, Type, cast
from pip._vendor.rich.color import ColorSystem
from pip._vendor.rich.style import Style
def move_cursor_forward(self) -> None:
    """Move the cursor forward a single cell. Wrap to the next line if required."""
    row, col = self.cursor_position
    if col == self.screen_size.col - 1:
        row += 1
        col = 0
    else:
        col += 1
    SetConsoleCursorPosition(self._handle, coords=WindowsCoordinates(row=row, col=col))