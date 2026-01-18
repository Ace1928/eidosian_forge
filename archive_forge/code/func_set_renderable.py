import sys
from typing import Optional, Tuple
from ._loop import loop_last
from .console import Console, ConsoleOptions, RenderableType, RenderResult
from .control import Control
from .segment import ControlType, Segment
from .style import StyleType
from .text import Text
def set_renderable(self, renderable: RenderableType) -> None:
    """Set a new renderable.

        Args:
            renderable (RenderableType): Any renderable object, including str.
        """
    self.renderable = renderable