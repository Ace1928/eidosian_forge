from __future__ import annotations
import re
from typing import Callable
from twisted.conch.insults import helper
from twisted.conch.insults.insults import (
from twisted.python import failure
from twisted.trial import unittest
def testEraseLine(self) -> None:
    s1 = b'line 1'
    s2 = b'line 2'
    s3 = b'line 3'
    self.term.write(b'\n'.join((s1, s2, s3)) + b'\n')
    self.term.cursorPosition(1, 1)
    self.term.eraseLine()
    self.assertEqual(self.term.__bytes__(), s1 + b'\n' + b'\n' + s3 + b'\n' + b'\n' * (HEIGHT - 4))