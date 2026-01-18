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
def test_invalidBase64EncodedResponse(self):
    """
        Test that a response which is not properly Base64 encoded results in
        the appropriate error code.
        """
    loginArgs = []
    self.server.portal = self.portalFactory(loginArgs)
    self.server.dataReceived(b'EHLO\r\n')
    self.server.dataReceived(b'AUTH LOGIN\r\n')
    self.assertServerResponse(b'x\r\n', [b'501 Syntax error in parameters or arguments'])
    self.assertEqual(loginArgs, [])