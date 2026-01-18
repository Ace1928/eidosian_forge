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
def test_lookupRecordTypeDefault(self):
    """
        L{Message.lookupRecordType} returns C{dns.UnknownRecord} if it is
        called with an integer which doesn't correspond to any known record
        type.
        """
    self.assertIs(dns.Message().lookupRecordType(65280), dns.UnknownRecord)