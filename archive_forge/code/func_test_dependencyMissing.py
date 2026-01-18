import datetime
import itertools
import sys
from unittest import skipIf
from zope.interface import implementer
from incremental import Version
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet._idna import _idnaText
from twisted.internet.error import CertificateError, ConnectionClosed, ConnectionLost
from twisted.internet.task import Clock
from twisted.python.compat import nativeString
from twisted.python.filepath import FilePath
from twisted.python.modules import getModule
from twisted.python.reflect import requireModule
from twisted.test.iosim import connectedServerAndClient
from twisted.test.test_twisted import SetAsideModule
from twisted.trial import util
from twisted.trial.unittest import SkipTest, SynchronousTestCase, TestCase
def test_dependencyMissing(self):
    """
        If I{service_identity} cannot be imported then
        L{_selectVerifyImplementation} returns L{simpleVerifyHostname} and
        L{SimpleVerificationError}.
        """
    with SetAsideModule('service_identity'):
        sys.modules['service_identity'] = None
        result = sslverify._selectVerifyImplementation()
        expected = (sslverify.simpleVerifyHostname, sslverify.simpleVerifyIPAddress, sslverify.SimpleVerificationError)
        self.assertEqual(expected, result)