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
def test_md(self):
    """
        Two L{dns.Record_MD} instances compare equal if and only if they have
        the same name and TTL.
        """
    self._simpleEqualityTest(dns.Record_MD)