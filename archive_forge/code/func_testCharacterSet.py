import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def testCharacterSet(self):
    self.parser.dataReceived(b''.join([b''.join([b'\x1b' + g + n for n in iterbytes(b'AB012')]) for g in iterbytes(b'()')]))
    occs = occurrences(self.proto)
    for which in (G0, G1):
        for charset in (CS_UK, CS_US, CS_DRAWING, CS_ALTERNATE, CS_ALTERNATE_SPECIAL):
            result = self.assertCall(occs.pop(0), 'selectCharacterSet', (charset, which))
            self.assertFalse(occurrences(result))
    self.assertFalse(occs)