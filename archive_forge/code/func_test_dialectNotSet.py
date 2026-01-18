import sys
from functools import partial
from io import BytesIO
from twisted.internet import main, protocol
from twisted.internet.testing import StringTransport
from twisted.python import failure
from twisted.python.compat import iterbytes
from twisted.spread import banana
from twisted.trial.unittest import TestCase
def test_dialectNotSet(self):
    """
        If no dialect has been selected and a PB VOCAB item is received,
        L{NotImplementedError} is raised.
        """
    self.assertRaises(NotImplementedError, self.enc.dataReceived, self.legalPbItem)