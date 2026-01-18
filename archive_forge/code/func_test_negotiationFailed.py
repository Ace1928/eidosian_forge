import datetime
import decimal
from typing import ClassVar, Dict, Type, TypeVar
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.internet import address, defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.protocols import amp
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.test import iosim
from twisted.trial.unittest import TestCase
def test_negotiationFailed(self):
    """
        Verify that starting TLS and failing on both sides at handshaking sends
        notifications to all the right places and terminates the connection.
        """
    badCert = GrumpyCert()
    cli, svr, p = connectedServerAndClient(ServerClass=SecurableProto, ClientClass=SecurableProto)
    svr.certFactory = lambda: badCert
    cli.callRemote(amp.StartTLS, tls_localCertificate=badCert)
    p.flush()
    self.assertEqual(badCert.verifyCount, 2)
    d = cli.callRemote(SecuredPing)
    p.flush()
    self.assertFailure(d, iosim.NativeOpenSSLError)