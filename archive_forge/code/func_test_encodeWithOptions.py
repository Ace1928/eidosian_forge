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
def test_encodeWithOptions(self):
    """
        L{dns._OPTHeader.options} is a list of L{dns._OPTVariableOption}
        instances which are packed into the rdata area of the header.
        """
    h = OPTNonStandardAttributes.object()
    h.options = [dns._OPTVariableOption(1, b'foobarbaz'), dns._OPTVariableOption(2, b'qux')]
    b = BytesIO()
    h.encode(b)
    self.assertEqual(b.getvalue(), OPTNonStandardAttributes.bytes(excludeOptions=True) + b'\x00\x14\x00\x01\x00\tfoobarbaz\x00\x02\x00\x03qux')