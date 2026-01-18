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
def test_fromStrCallsMessageFactory(self):
    """
        L{dns._EDNSMessage.fromString} calls L{dns._EDNSMessage._messageFactory}
        to create a new L{dns.Message} instance which is used to decode the
        supplied bytes.
        """

    class FakeMessageFactory:
        """
            Fake message factory.
            """

        def fromStr(self, *args, **kwargs):
            """
                Fake fromStr method which raises the arguments it was passed.

                @param args: positional arguments
                @param kwargs: keyword arguments
                """
            raise RaisedArgs(args, kwargs)
    m = dns._EDNSMessage()
    m._messageFactory = FakeMessageFactory
    dummyBytes = object()
    e = self.assertRaises(RaisedArgs, m.fromStr, dummyBytes)
    self.assertEqual(((dummyBytes,), {}), (e.args, e.kwargs))