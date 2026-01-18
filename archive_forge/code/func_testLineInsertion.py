import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def testLineInsertion(self):
    self.parser.dataReceived(b'\x1b[L\x1b[3L')
    occs = occurrences(self.proto)
    for arg in (1, 3):
        result = self.assertCall(occs.pop(0), 'insertLine', (arg,))
        self.assertFalse(occurrences(result))
    self.assertFalse(occs)