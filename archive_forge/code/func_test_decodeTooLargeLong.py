import sys
from functools import partial
from io import BytesIO
from twisted.internet import main, protocol
from twisted.internet.testing import StringTransport
from twisted.python import failure
from twisted.python.compat import iterbytes
from twisted.spread import banana
from twisted.trial.unittest import TestCase
def test_decodeTooLargeLong(self):
    """
        Test that a long above the implementation specific limit is rejected
        as too large to be decoded.
        """
    smallest = self._getSmallest()
    self.enc.setPrefixLimit(self.enc.prefixLimit * 2)
    self.enc.sendEncoded(smallest)
    encoded = self.io.getvalue()
    self.io.truncate(0)
    self.enc.setPrefixLimit(self.enc.prefixLimit // 2)
    self.assertRaises(banana.BananaError, self.enc.dataReceived, encoded)