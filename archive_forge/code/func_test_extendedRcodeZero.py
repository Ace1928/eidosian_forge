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
def test_extendedRcodeZero(self):
    """
        Note that EXTENDED-RCODE value 0 indicates that an unextended RCODE is
        in use (values 0 through 15).

        https://tools.ietf.org/html/rfc6891#section-6.1.3
        """
    ednsMessage = self.messageFactory(rCode=15, ednsVersion=0)
    standardMessage = ednsMessage._toMessage()
    self.assertEqual((15, 0), (standardMessage.rCode, standardMessage.additional[0].extendedRCODE))