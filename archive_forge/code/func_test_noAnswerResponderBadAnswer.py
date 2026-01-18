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
def test_noAnswerResponderBadAnswer(self):
    """
        Verify that responders of requiresAnswer=False commands have to return
        a dictionary anyway.

        (requiresAnswer is a hint from the _client_ - the server may be called
        upon to answer commands in any case, if the client wants to know when
        they complete.)
        """
    c, s, p = connectedServerAndClient(ServerClass=BadNoAnswerCommandProtocol, ClientClass=SimpleSymmetricCommandProtocol)
    c.callRemote(NoAnswerHello, hello=b'hello')
    p.flush()
    le = self.flushLoggedErrors(amp.BadLocalReturn)
    self.assertEqual(len(le), 1)