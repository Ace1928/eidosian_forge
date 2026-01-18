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
def test_hours(self):
    """
        Like C{test_seconds}, but for the C{"H"} suffix which multiplies the
        time value by C{3600}, the number of seconds in an hour.
        """
    self.assertEqual(3 * 3600, dns.str2time('3H'))