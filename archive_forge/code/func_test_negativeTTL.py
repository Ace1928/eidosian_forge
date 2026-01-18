import struct
from io import BytesIO
from zope.interface.verify import verifyClass
from twisted.internet import address, task
from twisted.internet.error import CannotListenError, ConnectionDone
from twisted.names import dns
from twisted.python.failure import Failure
from twisted.python.util import FancyEqMixin, FancyStrMixin
from twisted.test import proto_helpers
from twisted.test.testutils import ComparisonTestsMixin
from twisted.trial import unittest
def test_negativeTTL(self):
    """
        Attempting to create a L{dns.RRHeader} instance with a negative TTL
        causes L{ValueError} to be raised.
        """
    self.assertRaises(ValueError, dns.RRHeader, 'example.com', dns.A, dns.IN, -1, dns.Record_A('127.0.0.1'))