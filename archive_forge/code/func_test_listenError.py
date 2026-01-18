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
def test_listenError(self):
    """
        Exception L{CannotListenError} raised by C{listenUDP} should be turned
        into a C{Failure} passed to errback of the C{Deferred} returned by
        L{DNSDatagramProtocol.query}.
        """

    def startListeningError():
        raise CannotListenError(None, None, None)
    self.proto.startListening = startListeningError
    self.proto.transport = None
    d = self.proto.query(('127.0.0.1', 21345), [dns.Query(b'foo')])
    return self.assertFailure(d, CannotListenError)