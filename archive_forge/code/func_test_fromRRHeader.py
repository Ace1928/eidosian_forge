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
def test_fromRRHeader(self):
    """
        L{_OPTHeader.fromRRHeader} accepts an L{RRHeader} instance and
        returns an L{_OPTHeader} instance whose attribute values have
        been derived from the C{cls}, C{ttl} and C{payload} attributes
        of the original header.
        """
    genericHeader = dns.RRHeader(b'example.com', type=dns.OPT, cls=65535, ttl=254 << 24 | 253 << 16 | True << 15, payload=dns.UnknownRecord(b'\xff\xff\x00\x03abc'))
    decodedOptHeader = dns._OPTHeader.fromRRHeader(genericHeader)
    expectedOptHeader = dns._OPTHeader(udpPayloadSize=65535, extendedRCODE=254, version=253, dnssecOK=True, options=[dns._OPTVariableOption(code=65535, data=b'abc')])
    self.assertEqual(decodedOptHeader, expectedOptHeader)