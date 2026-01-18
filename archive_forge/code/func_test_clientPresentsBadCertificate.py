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
def test_clientPresentsBadCertificate(self):
    """
        When the server verifies and the client presents an invalid certificate
        for that verification by passing it to
        L{sslverify.optionsForClientTLS}, the connection cannot be established
        with an SSL error.
        """
    cProto, sProto, cWrapped, sWrapped, pump = self.serviceIdentitySetup('valid.example.com', 'valid.example.com', validCertificate=True, serverVerifies=True, validClientCertificate=False, clientPresentsCertificate=True)
    self.assertEqual(cWrapped.data, b'')
    cErr = cWrapped.lostReason.value
    sErr = sWrapped.lostReason.value
    self.assertIsInstance(cErr, SSL.Error)
    self.assertIsInstance(sErr, SSL.Error)