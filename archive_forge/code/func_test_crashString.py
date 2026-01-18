import sys
from functools import partial
from io import BytesIO
from twisted.internet import main, protocol
from twisted.internet.testing import StringTransport
from twisted.python import failure
from twisted.python.compat import iterbytes
from twisted.spread import banana
from twisted.trial.unittest import TestCase
def test_crashString(self):
    crashString = b'\x00\x00\x00\x00\x04\x80'
    try:
        self.enc.dataReceived(crashString)
    except banana.BananaError:
        pass