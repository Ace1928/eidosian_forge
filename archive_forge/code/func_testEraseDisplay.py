from __future__ import annotations
import re
from typing import Callable
from twisted.conch.insults import helper
from twisted.conch.insults.insults import (
from twisted.python import failure
from twisted.trial import unittest
def testEraseDisplay(self) -> None:
    self.term.write(b'Hello world\n')
    self.term.write(b'Goodbye world\n')
    self.term.eraseDisplay()
    self.assertEqual(self.term.__bytes__(), b'\n' * (HEIGHT - 1))