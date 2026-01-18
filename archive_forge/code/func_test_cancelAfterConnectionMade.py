import base64
import inspect
import re
from io import BytesIO
from typing import Any, List, Optional, Tuple, Type
from zope.interface import directlyProvides, implementer
import twisted.cred.checkers
import twisted.cred.credentials
import twisted.cred.error
import twisted.cred.portal
from twisted import cred
from twisted.cred.checkers import AllowAnonymousAccess, ICredentialsChecker
from twisted.cred.credentials import IAnonymous
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm, Portal
from twisted.internet import address, defer, error, interfaces, protocol, reactor, task
from twisted.internet.testing import MemoryReactor, StringTransport
from twisted.mail import smtp
from twisted.mail._cred import LOGINCredentials
from twisted.protocols import basic, loopback
from twisted.python.util import LineLog
from twisted.trial.unittest import TestCase
def test_cancelAfterConnectionMade(self):
    """
        When a user cancels L{twisted.mail.smtp.sendmail} after the connection
        is made, the connection is closed by
        L{twisted.internet.interfaces.ITransport.abortConnection}.
        """
    reactor = MemoryReactor()
    transport = AbortableStringTransport()
    d = smtp.sendmail('localhost', 'source@address', 'recipient@address', b'message', reactor=reactor)
    factory = reactor.tcpClients[0][2]
    p = factory.buildProtocol(None)
    p.makeConnection(transport)
    d.cancel()
    self.assertEqual(transport.aborting, True)
    self.assertEqual(transport.disconnecting, True)
    failure = self.failureResultOf(d)
    failure.trap(defer.CancelledError)