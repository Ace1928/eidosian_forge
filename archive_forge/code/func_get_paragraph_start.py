from __future__ import annotations
import re
import sys
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING, Optional
from pyglet import event
from pyglet.text import runlist
def get_paragraph_start(self, pos):
    """Get the starting position of a paragraph.

        :Parameters:
            `pos` : int
                Character position within paragraph.

        :rtype: int
        """
    if self._text[:pos + 1].endswith('\n') or self._text[:pos + 1].endswith(u'\u2029'):
        return pos
    m = self._previous_paragraph_re.search(self._text, 0, pos + 1)
    if not m:
        return 0
    return m.start() + 1