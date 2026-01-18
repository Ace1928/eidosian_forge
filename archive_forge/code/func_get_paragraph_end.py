from __future__ import annotations
import re
import sys
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING, Optional
from pyglet import event
from pyglet.text import runlist
def get_paragraph_end(self, pos):
    """Get the end position of a paragraph.

        :Parameters:
            `pos` : int
                Character position within paragraph.

        :rtype: int
        """
    m = self._next_paragraph_re.search(self._text, pos)
    if not m:
        return len(self._text)
    return m.start() + 1