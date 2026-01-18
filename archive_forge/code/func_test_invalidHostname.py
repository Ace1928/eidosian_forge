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
def test_invalidHostname(self):
    """
        When a certificate containing an invalid hostname is received from the
        server, the connection is immediately dropped.
        """
    cProto, sProto, cWrapped, sWrapped, pump = self.serviceIdentitySetup('wrong-host.example.com', 'correct-host.example.com')
    self.assertEqual(cWrapped.data, b'')
    self.assertEqual(sWrapped.data, b'')
    cErr = cWrapped.lostReason.value
    sErr = sWrapped.lostReason.value
    self.assertIsInstance(cErr, VerificationError)
    self.assertIsInstance(sErr, ConnectionClosed)