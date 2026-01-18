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
def test_bothWithTrailingDot(self):
    """
        L{dns._isSubdomainOf} returns C{True} if the first name
        is a subdomain of the second name and both names have a
        trailing ".".
        """
    assertIsSubdomainOf(self, b'foo.example.com.', b'example.com.')