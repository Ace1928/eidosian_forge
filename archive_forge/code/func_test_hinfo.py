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
def test_hinfo(self):
    """
        Two L{dns.Record_HINFO} instances compare equal if and only if they
        have the same cpu, os, and ttl.
        """
    self._equalityTest(dns.Record_HINFO('x86-64', 'plan9', 10), dns.Record_HINFO('x86-64', 'plan9', 10), dns.Record_HINFO('i386', 'plan9', 10))
    self._equalityTest(dns.Record_HINFO('x86-64', 'plan9', 10), dns.Record_HINFO('x86-64', 'plan9', 10), dns.Record_HINFO('x86-64', 'plan11', 10))
    self._equalityTest(dns.Record_HINFO('x86-64', 'plan9', 10), dns.Record_HINFO('x86-64', 'plan9', 10), dns.Record_HINFO('x86-64', 'plan9', 100))