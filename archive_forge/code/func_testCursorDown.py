from __future__ import annotations
import re
from typing import Callable
from twisted.conch.insults import helper
from twisted.conch.insults.insults import (
from twisted.python import failure
from twisted.trial import unittest
def testCursorDown(self) -> None:
    self.term.cursorDown(3)
    self.assertEqual(self.term.reportCursorPosition(), (0, 3))
    self.term.cursorDown()
    self.assertEqual(self.term.reportCursorPosition(), (0, 4))
    self.term.cursorDown(HEIGHT)
    self.assertEqual(self.term.reportCursorPosition(), (0, HEIGHT - 1))