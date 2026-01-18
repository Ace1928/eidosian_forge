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
def test_naptr(self):
    """
        Two L{dns.Record_NAPTR} instances compare equal if and only if they
        have the same order, preference, flags, service, regexp, replacement,
        and ttl.
        """
    self._equalityTest(dns.Record_NAPTR(1, 2, b'u', b'sip+E2U', b'/foo/bar/', b'baz', 12), dns.Record_NAPTR(1, 2, b'u', b'sip+E2U', b'/foo/bar/', b'baz', 12), dns.Record_NAPTR(2, 2, b'u', b'sip+E2U', b'/foo/bar/', b'baz', 12))
    self._equalityTest(dns.Record_NAPTR(1, 2, b'u', b'sip+E2U', b'/foo/bar/', b'baz', 12), dns.Record_NAPTR(1, 2, b'u', b'sip+E2U', b'/foo/bar/', b'baz', 12), dns.Record_NAPTR(1, 3, b'u', b'sip+E2U', b'/foo/bar/', b'baz', 12))
    self._equalityTest(dns.Record_NAPTR(1, 2, b'u', b'sip+E2U', b'/foo/bar/', b'baz', 12), dns.Record_NAPTR(1, 2, b'u', b'sip+E2U', b'/foo/bar/', b'baz', 12), dns.Record_NAPTR(1, 2, b'p', b'sip+E2U', b'/foo/bar/', b'baz', 12))
    self._equalityTest(dns.Record_NAPTR(1, 2, b'u', b'sip+E2U', b'/foo/bar/', b'baz', 12), dns.Record_NAPTR(1, 2, b'u', b'sip+E2U', b'/foo/bar/', b'baz', 12), dns.Record_NAPTR(1, 2, b'u', b'http', b'/foo/bar/', b'baz', 12))
    self._equalityTest(dns.Record_NAPTR(1, 2, b'u', b'sip+E2U', b'/foo/bar/', b'baz', 12), dns.Record_NAPTR(1, 2, b'u', b'sip+E2U', b'/foo/bar/', b'baz', 12), dns.Record_NAPTR(1, 2, b'u', b'sip+E2U', b'/bar/foo/', b'baz', 12))
    self._equalityTest(dns.Record_NAPTR(1, 2, b'u', b'sip+E2U', b'/foo/bar/', b'baz', 12), dns.Record_NAPTR(1, 2, b'u', b'sip+E2U', b'/foo/bar/', b'baz', 12), dns.Record_NAPTR(1, 2, b'u', b'sip+E2U', b'/bar/foo/', b'quux', 12))
    self._equalityTest(dns.Record_NAPTR(1, 2, b'u', b'sip+E2U', b'/foo/bar/', b'baz', 12), dns.Record_NAPTR(1, 2, b'u', b'sip+E2U', b'/foo/bar/', b'baz', 12), dns.Record_NAPTR(1, 2, b'u', b'sip+E2U', b'/bar/foo/', b'baz', 5))