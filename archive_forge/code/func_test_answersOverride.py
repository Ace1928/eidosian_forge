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
def test_answersOverride(self):
    """
        L{dns._EDNSMessage.answers} can be overridden in the constructor.
        """
    msg = self.messageFactory(answers=[dns.RRHeader(b'example.com', payload=dns.Record_A('1.2.3.4'))])
    self.assertEqual(msg.answers, [dns.RRHeader(b'example.com', payload=dns.Record_A('1.2.3.4'))])