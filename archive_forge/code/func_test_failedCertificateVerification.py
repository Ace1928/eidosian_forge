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
def test_failedCertificateVerification(self):
    """
        Check that connecting with a certificate not accepted by the server CA
        fails.
        """
    onServerLost = defer.Deferred()
    onClientLost = defer.Deferred()
    self.loopback(sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, verify=False, requireCertificate=False), sslverify.OpenSSLCertificateOptions(verify=True, requireCertificate=False, caCerts=[self.cCert]), onServerLost=onServerLost, onClientLost=onClientLost)
    d = defer.DeferredList([onClientLost, onServerLost], consumeErrors=True)

    def afterLost(result):
        (cSuccess, cResult), (sSuccess, sResult) = result
        self.assertFalse(cSuccess)
        self.assertFalse(sSuccess)
    return d.addCallback(afterLost)