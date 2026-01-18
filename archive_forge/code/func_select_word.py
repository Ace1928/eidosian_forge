from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Any, Optional, Pattern
import re
import time
from pyglet import clock
from pyglet import event
from pyglet.window import key
def select_word(self, x: int, y: int) -> None:
    """Select the word at the given window coordinate.

        :Parameters:
            `x` : int   
                X coordinate.
            `y` : int
                Y coordinate.

        """
    line = self._layout.get_line_from_point(x, y)
    p = self._layout.get_position_on_line(line, x)
    match1 = self._previous_word_re.search(self._layout.document.text, 0, p + 1)
    if not match1:
        mark1 = 0
    else:
        mark1 = match1.start()
    self.mark = mark1
    match2 = self._next_word_re.search(self._layout.document.text, p)
    if not match2:
        mark2 = len(self._layout.document.text)
    else:
        mark2 = match2.start()
    self._position = mark2
    self._update(line=line)
    self._next_attributes.clear()