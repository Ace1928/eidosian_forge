import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def test_cursorHome(self):
    """
        L{ServerProtocol.cursorHome} writes a control sequence ending
        with L{CSFinalByte.CUP} and no parameters, so that the client
        defaults to (1, 1).
        """
    self.protocol.cursorHome()
    self.assertEqual(self.transport.value(), self.CSI + CSFinalByte.CUP.value)