from __future__ import annotations
import re
from typing import Callable
from twisted.conch.insults import helper
from twisted.conch.insults.insults import (
from twisted.python import failure
from twisted.trial import unittest
def testSynchronous(self) -> None:
    self.term.write(b'hello world')
    result: list[re.Match[bytes]] = []
    d = self.term.expect(b'hello world')
    d.addCallback(result.append)
    self.assertTrue(result)
    self.assertEqual(result[0].group(), b'hello world')