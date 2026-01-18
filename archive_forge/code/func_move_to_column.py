import sys
import time
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Union
from .segment import ControlCode, ControlType, Segment
@classmethod
def move_to_column(cls, x: int, y: int=0) -> 'Control':
    """Move to the given column, optionally add offset to row.

        Returns:
            x (int): absolute x (column)
            y (int): optional y offset (row)

        Returns:
            ~Control: Control object.
        """
    return cls((ControlType.CURSOR_MOVE_TO_COLUMN, x), (ControlType.CURSOR_DOWN if y > 0 else ControlType.CURSOR_UP, abs(y))) if y else cls((ControlType.CURSOR_MOVE_TO_COLUMN, x))