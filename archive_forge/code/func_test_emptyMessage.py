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
def test_emptyMessage(self):
    """
        Test that a message which has been truncated causes an EOFError to
        be raised when it is parsed.
        """
    msg = dns.Message()
    self.assertRaises(EOFError, msg.fromStr, b'')