import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def testSingleShifts(self):
    self.parser.dataReceived(b'\x1bN\x1bO')
    occs = occurrences(self.proto)
    result = self.assertCall(occs.pop(0), 'singleShift2')
    self.assertFalse(occurrences(result))
    result = self.assertCall(occs.pop(0), 'singleShift3')
    self.assertFalse(occurrences(result))
    self.assertFalse(occs)