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
def test_nonAuthoritativeMessageEncode(self):
    """
        If the message C{authoritative} attribute is set to 0, the encoded bytes
        will have AA bit 0.
        """
    self.assertEqual(self.messageFactory(**MessageNonAuthoritative.kwargs()).toStr(), MessageNonAuthoritative.bytes())