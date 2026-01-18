from __future__ import annotations
import re
from typing import Callable
from twisted.conch.insults import helper
from twisted.conch.insults.insults import (
from twisted.python import failure
from twisted.trial import unittest
def testSimpleWriting(self) -> None:
    s = b'Hello, world.'
    self.term.write(s)
    self.assertEqual(self.term.__bytes__(), s + b'\n' + b'\n' * (HEIGHT - 2))