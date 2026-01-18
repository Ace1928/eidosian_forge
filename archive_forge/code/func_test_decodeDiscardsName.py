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
def test_decodeDiscardsName(self):
    """
        L{dns._OPTHeader.decode} discards the name which is encoded in
        the supplied bytes. The name attribute of the resulting
        L{dns._OPTHeader} instance will always be L{dns.Name(b'')}.
        """
    b = BytesIO(OPTNonStandardAttributes.bytes(excludeName=True) + b'\x07example\x03com\x00')
    h = dns._OPTHeader()
    h.decode(b)
    self.assertEqual(h.name, dns.Name(b''))