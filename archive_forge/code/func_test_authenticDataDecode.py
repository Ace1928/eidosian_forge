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
def test_authenticDataDecode(self):
    """
        L{dns.Message.fromStr} decodes byte4 and assigns bit3 to
        L{dns.Message.authenticData}.
        """
    m = dns.Message()
    m.fromStr(MESSAGE_AUTHENTIC_DATA_BYTES)
    self.assertEqual(m.authenticData, 1)