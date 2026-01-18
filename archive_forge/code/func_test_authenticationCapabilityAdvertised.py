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
def test_authenticationCapabilityAdvertised(self):
    """
        Test that AUTH is advertised to clients which issue an EHLO command.
        """
    self.transport.clear()
    self.server.dataReceived(b'EHLO\r\n')
    responseLines = self.transport.value().splitlines()
    self.assertEqual(responseLines[0], b'250-localhost Hello 127.0.0.1, nice to meet you')
    self.assertEqual(responseLines[1], b'250 AUTH LOGIN')
    self.assertEqual(len(responseLines), 2)