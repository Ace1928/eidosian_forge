from __future__ import annotations
import re
from typing import Callable
from twisted.conch.insults import helper
from twisted.conch.insults.insults import (
from twisted.python import failure
from twisted.trial import unittest
def testMultiple(self) -> None:
    result: list[re.Match[bytes]] = []
    d1 = self.term.expect(b'hello ')
    d1.addCallback(result.append)
    d2 = self.term.expect(b'world')
    d2.addCallback(result.append)
    self.assertFalse(result)
    self.term.write(b'hello')
    self.assertFalse(result)
    self.term.write(b' ')
    self.assertEqual(len(result), 1)
    self.term.write(b'world')
    self.assertEqual(len(result), 2)
    self.assertEqual(result[0].group(), b'hello ')
    self.assertEqual(result[1].group(), b'world')