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
def test_decodeRdlengthTooShort(self):
    """
        L{dns._OPTHeader.decode} raises an exception if the supplied
        RDLEN is too short.
        """
    b = BytesIO(OPTNonStandardAttributes.bytes(excludeOptions=True) + b'\x00\x05\x00\x01\x00\x02\x00\x00')
    h = dns._OPTHeader()
    self.assertRaises(EOFError, h.decode, b)