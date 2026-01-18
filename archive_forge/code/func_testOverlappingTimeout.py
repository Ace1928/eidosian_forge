from __future__ import annotations
import re
from typing import Callable
from twisted.conch.insults import helper
from twisted.conch.insults.insults import (
from twisted.python import failure
from twisted.trial import unittest
def testOverlappingTimeout(self) -> None:
    self.term.write(b'not zoomtastic')
    result: list[re.Match[bytes]] = []
    d1 = self.term.expect(b'hello world', timeout=1, scheduler=self.fs)
    d1.addBoth(self._cbTestTimeoutFailure)
    d2 = self.term.expect(b'zoom')
    d2.addCallback(result.append)
    self.fs.calls[0].call()
    self.assertEqual(len(result), 1)
    self.assertEqual(result[0].group(), b'zoom')