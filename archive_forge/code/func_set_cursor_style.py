from __future__ import annotations
import abc
import time
import typing
from .common import BaseScreen
def set_cursor_style(self, style: Literal[1, 2, 3, 4]) -> None:
    """
        style -- CURSOR_BLINKING_BLOCK, CURSOR_UNDERSCORE,
            CURSOR_BLINKING_BLOCK_UNDERSCORE or
            CURSOR_INVERTING_BLINKING_BLOCK
        """
    if not 1 <= style <= 4:
        raise ValueError(style)
    self.cursor_style = style
    self._update_cursor = True