import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def test_applicationDataBytes(self):
    """
        Contiguous non-control bytes are passed to a single call to the
        C{write} method of the terminal to which the L{ClientProtocol} is
        connected.
        """
    occs = occurrences(self.proto)
    self.parser.dataReceived(b'a')
    self.assertCall(occs.pop(0), 'write', (b'a',))
    self.parser.dataReceived(b'bc')
    self.assertCall(occs.pop(0), 'write', (b'bc',))