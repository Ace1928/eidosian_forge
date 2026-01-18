import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def testShifting(self):
    self.parser.dataReceived(b'\x15\x14')
    occs = occurrences(self.proto)
    result = self.assertCall(occs.pop(0), 'shiftIn')
    self.assertFalse(occurrences(result))
    result = self.assertCall(occs.pop(0), 'shiftOut')
    self.assertFalse(occurrences(result))
    self.assertFalse(occs)