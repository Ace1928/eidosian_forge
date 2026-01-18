from __future__ import annotations
import re
from typing import Callable
from twisted.conch.insults import helper
from twisted.conch.insults.insults import (
from twisted.python import failure
from twisted.trial import unittest
def testWritingInTheMiddle(self) -> None:
    s = b'Hello, world.'
    self.term.cursorDown(5)
    self.term.cursorForward(5)
    self.term.write(s)
    self.assertEqual(self.term.__bytes__(), b'\n' * 5 + self.term.fill * 5 + s + b'\n' + b'\n' * (HEIGHT - 7))