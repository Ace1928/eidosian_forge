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
def test_constructorArguments(self):
    """
        L{dns._OPTVariableOption.__init__} requires code and data
        arguments which are saved as public instance attributes.
        """
    h = dns._OPTVariableOption(1, b'x')
    self.assertEqual(h.code, 1)
    self.assertEqual(h.data, b'x')