import sys
from functools import partial
from io import BytesIO
from twisted.internet import main, protocol
from twisted.internet.testing import StringTransport
from twisted.python import failure
from twisted.python.compat import iterbytes
from twisted.spread import banana
from twisted.trial.unittest import TestCase
def test_integer(self):
    self.enc.sendEncoded(1015)
    self.enc.dataReceived(self.io.getvalue())
    self.assertEqual(self.result, 1015)