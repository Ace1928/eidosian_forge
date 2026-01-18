import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def test_eraseToLineBeginning(self):
    """
        L{ServerProtocol.eraseToLineBeginning} writes the control
        sequence sequence ending in L{CSFinalByte.EL} and a parameter
        of 1 (from the beginning of the line up to and include the
        active present position's current location.)
        """
    self.protocol.eraseToLineBeginning()
    self.assertEqual(self.transport.value(), self.CSI + b'1' + CSFinalByte.EL.value)