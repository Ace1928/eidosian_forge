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
def test_decodeWithCompression(self):
    """
        If the leading byte of an encoded label (in bytes read from a stream
        passed to L{Name.decode}) has its two high bits set, the next byte is
        treated as a pointer to another label in the stream and that label is
        included in the name being decoded.
        """
    stream = BytesIO(b'x' * 20 + b'\x01f\x03isi\x04arpa\x00\x03foo\xc0\x14\x03bar\xc0 ')
    stream.seek(20)
    name = dns.Name()
    name.decode(stream)
    self.assertEqual(b'f.isi.arpa', name.name)
    self.assertEqual(32, stream.tell())
    name.decode(stream)
    self.assertEqual(name.name, b'foo.f.isi.arpa')
    self.assertEqual(38, stream.tell())
    name.decode(stream)
    self.assertEqual(name.name, b'bar.foo.f.isi.arpa')
    self.assertEqual(44, stream.tell())