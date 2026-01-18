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
def test_afsdb(self):
    """
        Two L{dns.Record_AFSDB} instances compare equal if and only if they
        have the same subtype, hostname, and ttl.
        """
    self._equalityTest(dns.Record_AFSDB(1, b'example.com', 2), dns.Record_AFSDB(1, b'example.com', 2), dns.Record_AFSDB(2, b'example.com', 2))
    self._equalityTest(dns.Record_AFSDB(1, b'example.com', 2), dns.Record_AFSDB(1, b'example.com', 2), dns.Record_AFSDB(1, b'example.org', 2))
    self._equalityTest(dns.Record_AFSDB(1, b'example.com', 2), dns.Record_AFSDB(1, b'example.com', 2), dns.Record_AFSDB(1, b'example.com', 3))