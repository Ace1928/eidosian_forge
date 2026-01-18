import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def test_eraseToDisplayBeginning(self):
    """
        L{ServerProtocol.eraseToDisplayBeginning} writes the control
        sequence sequence ending in L{CSFinalByte.ED} a parameter of 1
        (from the beginning of the page up to and include the active
        present position's current location.)
        """
    self.protocol.eraseToDisplayBeginning()
    self.assertEqual(self.transport.value(), self.CSI + b'1' + CSFinalByte.ED.value)