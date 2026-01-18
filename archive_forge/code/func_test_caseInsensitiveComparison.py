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
def test_caseInsensitiveComparison(self):
    """
        L{dns._isSubdomainOf} does case-insensitive comparison
        of name labels.
        """
    assertIsSubdomainOf(self, b'foo.example.com', b'EXAMPLE.COM')
    assertIsSubdomainOf(self, b'FOO.EXAMPLE.COM', b'example.com')