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
def test_encodeWithCompression(self):
    """
        If a compression dictionary is passed to it, L{Name.encode} uses offset
        information from it to encode its name with references to existing
        labels in the stream instead of including another copy of them in the
        output.  It also updates the compression dictionary with the location of
        the name it writes to the stream.
        """
    name = dns.Name(b'foo.example.com')
    compression = {b'example.com': 23}
    previous = b'some prefix to change .tell()'
    stream = BytesIO()
    stream.write(previous)
    expected = len(previous) + dns.Message.headerSize
    name.encode(stream, compression)
    self.assertEqual(b'\x03foo\xc0\x17', stream.getvalue()[len(previous):])
    self.assertEqual({b'example.com': 23, b'foo.example.com': expected}, compression)