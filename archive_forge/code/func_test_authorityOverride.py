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
def test_authorityOverride(self):
    """
        L{dns._EDNSMessage.authority} can be overridden in the constructor.
        """
    msg = self.messageFactory(authority=[dns.RRHeader(b'example.com', type=dns.SOA, payload=dns.Record_SOA())])
    self.assertEqual(msg.authority, [dns.RRHeader(b'example.com', type=dns.SOA, payload=dns.Record_SOA())])