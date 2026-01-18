from __future__ import annotations
import re
from typing import Callable
from twisted.conch.insults import helper
from twisted.conch.insults.insults import (
from twisted.python import failure
from twisted.trial import unittest
def testInitialState(self) -> None:
    self.assertEqual(self.term.width, WIDTH)
    self.assertEqual(self.term.height, HEIGHT)
    self.assertEqual(self.term.__bytes__(), b'\n' * (HEIGHT - 1))
    self.assertEqual(self.term.reportCursorPosition(), (0, 0))