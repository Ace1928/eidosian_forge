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
def assertIsSubdomainOf(testCase, descendant, ancestor):
    """
    Assert that C{descendant} *is* a subdomain of C{ancestor}.

    @type testCase: L{unittest.SynchronousTestCase}
    @param testCase: The test case on which to run the assertions.

    @type descendant: C{str}
    @param descendant: The subdomain name to test.

    @type ancestor: C{str}
    @param ancestor: The superdomain name to test.
    """
    testCase.assertTrue(dns._isSubdomainOf(descendant, ancestor), f'{descendant!r} is not a subdomain of {ancestor!r}')