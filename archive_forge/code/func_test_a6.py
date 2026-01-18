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
def test_a6(self):
    """
        Two L{dns.Record_A6} instances compare equal if and only if they have
        the same prefix, prefix length, suffix, and ttl.
        """
    self._equalityTest(dns.Record_A6(16, '::abcd', b'example.com', 10), dns.Record_A6(16, '::abcd', b'example.com', 10), dns.Record_A6(32, '::abcd', b'example.com', 10))
    self._equalityTest(dns.Record_A6(16, '::abcd', b'example.com', 10), dns.Record_A6(16, '::abcd', b'example.com', 10), dns.Record_A6(16, '::abcd:0', b'example.com', 10))
    self._equalityTest(dns.Record_A6(16, '::abcd', b'example.com', 10), dns.Record_A6(16, '::abcd', b'example.com', 10), dns.Record_A6(16, '::abcd', b'example.org', 10))
    self._equalityTest(dns.Record_A6(16, '::abcd', b'example.com', 10), dns.Record_A6(16, '::abcd', b'example.com', 10), dns.Record_A6(16, '::abcd', b'example.com', 100))