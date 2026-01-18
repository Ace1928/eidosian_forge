from __future__ import annotations
import re
from typing import Callable
from twisted.conch.insults import helper
from twisted.conch.insults.insults import (
from twisted.python import failure
from twisted.trial import unittest
def test_carriageReturn(self) -> None:
    """
        C{"\r"} moves the cursor to the first column in the current row.
        """
    self.term.cursorForward(5)
    self.term.cursorDown(3)
    self.assertEqual(self.term.reportCursorPosition(), (5, 3))
    self.term.insertAtCursor(b'\r')
    self.assertEqual(self.term.reportCursorPosition(), (0, 3))