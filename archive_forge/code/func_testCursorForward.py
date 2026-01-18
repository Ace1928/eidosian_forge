from __future__ import annotations
import re
from typing import Callable
from twisted.conch.insults import helper
from twisted.conch.insults.insults import (
from twisted.python import failure
from twisted.trial import unittest
def testCursorForward(self) -> None:
    self.term.cursorForward(2)
    self.assertEqual(self.term.reportCursorPosition(), (2, 0))
    self.term.cursorForward(2)
    self.assertEqual(self.term.reportCursorPosition(), (4, 0))
    self.term.cursorForward(WIDTH)
    self.assertEqual(self.term.reportCursorPosition(), (WIDTH, 0))