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
def test_helloFatalErrorHandling(self):
    """
        Verify that if a known, fatal error type is raised and handled, it will
        be properly relayed to the other end of the connection and translated
        into an exception, no error will be logged, and the connection will be
        terminated.
        """
    L = []
    c, s, p = connectedServerAndClient(ServerClass=SimpleSymmetricCommandProtocol, ClientClass=SimpleSymmetricCommandProtocol)
    HELLO = b'die'
    c.sendHello(HELLO).addErrback(L.append)
    p.flush()
    L.pop().trap(DeathThreat)
    c.sendHello(HELLO).addErrback(L.append)
    p.flush()
    L.pop().trap(error.ConnectionDone)