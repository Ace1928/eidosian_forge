import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def testModes(self):
    self.parser.dataReceived(b'\x1b[' + b';'.join((b'%d' % (m,) for m in [modes.KAM, modes.IRM, modes.LNM])) + b'h')
    self.parser.dataReceived(b'\x1b[' + b';'.join((b'%d' % (m,) for m in [modes.KAM, modes.IRM, modes.LNM])) + b'l')
    occs = occurrences(self.proto)
    result = self.assertCall(occs.pop(0), 'setModes', ([modes.KAM, modes.IRM, modes.LNM],))
    self.assertFalse(occurrences(result))
    result = self.assertCall(occs.pop(0), 'resetModes', ([modes.KAM, modes.IRM, modes.LNM],))
    self.assertFalse(occurrences(result))
    self.assertFalse(occs)