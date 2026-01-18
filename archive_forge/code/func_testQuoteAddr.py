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
def testQuoteAddr(self):
    cases = [[b'user@host.name', b'<user@host.name>'], [b'"User Name" <user@host.name>', b'<user@host.name>'], [smtp.Address(b'someguy@someplace'), b'<someguy@someplace>'], [b'', b'<>'], [smtp.Address(b''), b'<>']]
    for c, e in cases:
        self.assertEqual(smtp.quoteaddr(c), e)