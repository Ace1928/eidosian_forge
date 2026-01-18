import sys
from typing import Optional, Tuple
from ._loop import loop_last
from .console import Console, ConsoleOptions, RenderableType, RenderResult
from .control import Control
from .segment import ControlType, Segment
from .style import StyleType
from .text import Text
def restore_cursor(self) -> Control:
    """Get control codes to clear the render and restore the cursor to its previous position.

        Returns:
            Control: A Control instance that may be printed.
        """
    if self._shape is not None:
        _, height = self._shape
        return Control(ControlType.CARRIAGE_RETURN, *((ControlType.CURSOR_UP, 1), (ControlType.ERASE_IN_LINE, 2)) * height)
    return Control()