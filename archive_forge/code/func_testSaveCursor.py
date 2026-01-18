from __future__ import annotations
import re
from typing import Callable
from twisted.conch.insults import helper
from twisted.conch.insults.insults import (
from twisted.python import failure
from twisted.trial import unittest
def testSaveCursor(self) -> None:
    self.term.cursorDown(5)
    self.term.cursorForward(7)
    self.assertEqual(self.term.reportCursorPosition(), (7, 5))
    self.term.saveCursor()
    self.term.cursorDown(7)
    self.term.cursorBackward(3)
    self.assertEqual(self.term.reportCursorPosition(), (4, 12))
    self.term.restoreCursor()
    self.assertEqual(self.term.reportCursorPosition(), (7, 5))