from __future__ import annotations
import re
import sys
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING, Optional
from pyglet import event
from pyglet.text import runlist
def delete_text(self, start, end):
    """Delete text from the document.

        :Parameters:
            `start` : int
                Starting character position to delete from.
            `end` : int
                Ending character position to delete to (exclusive).

        """
    self._delete_text(start, end)
    self.dispatch_event('on_delete_text', start, end)