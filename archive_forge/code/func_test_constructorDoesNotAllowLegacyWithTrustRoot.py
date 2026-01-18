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
def test_constructorDoesNotAllowLegacyWithTrustRoot(self):
    """
        C{verify}, C{requireCertificate}, and C{caCerts} must not be specified
        by the caller (to be I{any} value, even the default!) when specifying
        C{trustRoot}.
        """
    self.assertRaises(TypeError, sslverify.OpenSSLCertificateOptions, privateKey=self.sKey, certificate=self.sCert, verify=True, trustRoot=None, caCerts=self.caCerts)
    self.assertRaises(TypeError, sslverify.OpenSSLCertificateOptions, privateKey=self.sKey, certificate=self.sCert, trustRoot=None, requireCertificate=True)