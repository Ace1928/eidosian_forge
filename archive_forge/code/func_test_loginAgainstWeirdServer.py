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
def test_loginAgainstWeirdServer(self):
    """
        When communicating with a server which implements the I{LOGIN} SASL
        mechanism using C{"Username:"} as the challenge (rather than C{"User
        Name\\0"}), L{ESMTPClient} can still authenticate successfully using
        the I{LOGIN} mechanism.
        """
    realm = DummyRealm()
    p = cred.portal.Portal(realm)
    p.registerChecker(DummyChecker())
    server = DummyESMTP({b'LOGIN': smtp.LOGINCredentials})
    server.portal = p
    client = MyESMTPClient(b'testpassword')
    cAuth = smtp.LOGINAuthenticator(b'testuser')
    client.registerAuthenticator(cAuth)
    d = self.loopback(server, client)
    d.addCallback(lambda x: self.assertTrue(server.authenticated))
    return d