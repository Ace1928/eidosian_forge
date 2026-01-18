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
def test_checkingDisabledDecode(self):
    """
        L{dns.Message.fromStr} decodes byte4 and assigns bit4 to
        L{dns.Message.checkingDisabled}.
        """
    m = dns.Message()
    m.fromStr(MESSAGE_CHECKING_DISABLED_BYTES)
    self.assertEqual(m.checkingDisabled, 1)