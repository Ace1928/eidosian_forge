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
def test_methodIsDeprecated(self):
    """
        Passing C{method} to L{sslverify.OpenSSLCertificateOptions} is
        deprecated.
        """
    sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, method=SSL.SSLv23_METHOD)
    message = 'Passing method to twisted.internet.ssl.CertificateOptions was deprecated in Twisted 17.1.0. Please use a combination of insecurelyLowerMinimumTo, raiseMinimumTo, and lowerMaximumSecurityTo instead, as Twisted will correctly configure the method.'
    warnings = self.flushWarnings([self.test_methodIsDeprecated])
    self.assertEqual(1, len(warnings))
    self.assertEqual(DeprecationWarning, warnings[0]['category'])
    self.assertEqual(message, warnings[0]['message'])