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
def assertEncodedFormat(self, expectedEncoding, record):
    """
        Assert that encoding C{record} produces the expected bytes.

        @type record: L{dns.IEncodable}
        @param record: A record instance to encode

        @type expectedEncoding: L{bytes}
        @param expectedEncoding: The value which C{record.encode()}
            should produce.
        """
    stream = BytesIO()
    record.encode(stream)
    self.assertEqual(stream.getvalue(), expectedEncoding)