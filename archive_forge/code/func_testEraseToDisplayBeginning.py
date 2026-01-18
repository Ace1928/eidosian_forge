from __future__ import annotations
import re
from typing import Callable
from twisted.conch.insults import helper
from twisted.conch.insults.insults import (
from twisted.python import failure
from twisted.trial import unittest
def testEraseToDisplayBeginning(self) -> None:
    s1 = b'Hello world'
    s2 = b'Goodbye world'
    self.term.write(b'\n'.join((s1, s2)))
    self.term.cursorPosition(5, 1)
    self.term.eraseToDisplayBeginning()
    self.assertEqual(self.term.__bytes__(), b'\n' + s2[6:].rjust(len(s2)) + b'\n' + b'\n' * (HEIGHT - 3))