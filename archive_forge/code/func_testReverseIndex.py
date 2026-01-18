from __future__ import annotations
import re
from typing import Callable
from twisted.conch.insults import helper
from twisted.conch.insults.insults import (
from twisted.python import failure
from twisted.trial import unittest
def testReverseIndex(self) -> None:
    self.term.reverseIndex()
    self.assertEqual(self.term.reportCursorPosition(), (0, 0))
    self.term.cursorDown(2)
    self.assertEqual(self.term.reportCursorPosition(), (0, 2))
    self.term.reverseIndex()
    self.assertEqual(self.term.reportCursorPosition(), (0, 1))