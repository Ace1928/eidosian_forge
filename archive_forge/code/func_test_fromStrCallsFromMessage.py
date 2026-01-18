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
def test_fromStrCallsFromMessage(self):
    """
        L{dns._EDNSMessage.fromString} calls L{dns._EDNSMessage._fromMessage}
        with a L{dns.Message} instance
        """
    m = dns._EDNSMessage()

    class FakeMessageFactory:
        """
            Fake message factory.
            """

        def fromStr(self, bytes):
            """
                A noop fake version of fromStr

                @param bytes: the bytes to be decoded
                """
    fakeMessage = FakeMessageFactory()
    m._messageFactory = lambda: fakeMessage

    def fakeFromMessage(*args, **kwargs):
        raise RaisedArgs(args, kwargs)
    m._fromMessage = fakeFromMessage
    e = self.assertRaises(RaisedArgs, m.fromStr, b'')
    self.assertEqual(((fakeMessage,), {}), (e.args, e.kwargs))