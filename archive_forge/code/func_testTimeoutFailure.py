from __future__ import annotations
import re
from typing import Callable
from twisted.conch.insults import helper
from twisted.conch.insults.insults import (
from twisted.python import failure
from twisted.trial import unittest
def testTimeoutFailure(self) -> None:
    d = self.term.expect(b'hello world', timeout=1, scheduler=self.fs)
    d.addBoth(self._cbTestTimeoutFailure)
    self.fs.calls[0].call()