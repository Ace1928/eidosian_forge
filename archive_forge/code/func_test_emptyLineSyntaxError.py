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
def test_emptyLineSyntaxError(self):
    """
        If L{smtp.SMTP} receives an empty line, it responds with a 500 error
        response code and a message about a syntax error.
        """
    proto = smtp.SMTP()
    transport = StringTransport()
    proto.makeConnection(transport)
    proto.lineReceived(b'')
    proto.setTimeout(None)
    out = transport.value().splitlines()
    self.assertEqual(len(out), 2)
    self.assertTrue(out[0].startswith(b'220'))
    self.assertEqual(out[1], b'500 Error: bad syntax')