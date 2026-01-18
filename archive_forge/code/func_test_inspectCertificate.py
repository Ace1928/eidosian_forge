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
def test_inspectCertificate(self):
    """
        Test that the C{inspect} method of L{sslverify.Certificate} returns
        a human-readable string containing some basic information about the
        certificate.
        """
    c = sslverify.Certificate.loadPEM(A_HOST_CERTIFICATE_PEM)
    pk = c.getPublicKey()
    keyHash = pk.keyHash()
    self.assertEqual(c.inspect().split('\n'), ['Certificate For Subject:', '               Common Name: example.twistedmatrix.com', '              Country Name: US', '             Email Address: nobody@twistedmatrix.com', '             Locality Name: Boston', '         Organization Name: Twisted Matrix Labs', '  Organizational Unit Name: Security', '    State Or Province Name: Massachusetts', '', 'Issuer:', '               Common Name: example.twistedmatrix.com', '              Country Name: US', '             Email Address: nobody@twistedmatrix.com', '             Locality Name: Boston', '         Organization Name: Twisted Matrix Labs', '  Organizational Unit Name: Security', '    State Or Province Name: Massachusetts', '', 'Serial Number: 12345', 'Digest: C4:96:11:00:30:C3:EC:EE:A3:55:AA:ED:8C:84:85:18', 'Public Key with Hash: ' + keyHash])