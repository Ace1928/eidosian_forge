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
def test_helloWorld(self):
    """
        Verify that a simple command can be sent and its response received with
        the simple low-level string-based API.
        """
    c, s, p = connectedServerAndClient()
    L = []
    HELLO = b'world'
    c.sendHello(HELLO).addCallback(L.append)
    p.flush()
    self.assertEqual(L[0][b'hello'], HELLO)