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
def test_messageStringMadeFile(self):
    """
        L{twisted.mail.smtp.sendmail} will turn non-file-like objects
        (eg. strings) into file-like objects before sending.
        """
    reactor = MemoryReactor()
    smtp.sendmail('localhost', 'source@address', 'recipient@address', b'message', reactor=reactor)
    factory = reactor.tcpClients[0][2]
    messageFile = factory.file
    messageFile.seek(0)
    self.assertEqual(messageFile.read(), b'message')