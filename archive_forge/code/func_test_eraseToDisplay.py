import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def test_eraseToDisplay(self):
    """
        L{ServerProtocol.eraseDisplay} writes the control sequence
        sequence ending in L{CSFinalByte.ED} a parameter of 2 (the
        entire page)
        """
    self.protocol.eraseDisplay()
    self.assertEqual(self.transport.value(), self.CSI + b'2' + CSFinalByte.ED.value)