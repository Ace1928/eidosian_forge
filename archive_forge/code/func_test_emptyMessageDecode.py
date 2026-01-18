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
def test_emptyMessageDecode(self):
    """
        An empty message byte sequence can be decoded.
        """
    m = self.messageFactory()
    m.fromStr(MessageEmpty.bytes())
    self.assertEqual(m, self.messageFactory(**MessageEmpty.kwargs()))