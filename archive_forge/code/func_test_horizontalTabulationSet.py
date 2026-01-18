import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def test_horizontalTabulationSet(self):
    """
        L{ServerProtocol.horizontalTabulationSet} writes the escape
        sequence ending in L{C1SevenBit.HTS}
        """
    self.protocol.horizontalTabulationSet()
    self.assertEqual(self.transport.value(), self.ESC + C1SevenBit.HTS.value)