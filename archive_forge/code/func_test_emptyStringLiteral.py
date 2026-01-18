from __future__ import annotations
import base64
import codecs
import functools
import locale
import os
import uuid
from collections import OrderedDict
from io import BytesIO
from itertools import chain
from typing import Optional, Type
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.cred.checkers import InMemoryUsernamePasswordDatabaseDontUse
from twisted.cred.credentials import (
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm, Portal
from twisted.internet import defer, error, interfaces, reactor
from twisted.internet.defer import Deferred
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.mail import imap4
from twisted.mail.imap4 import MessageSet
from twisted.mail.interfaces import (
from twisted.protocols import loopback
from twisted.python import failure, log, util
from twisted.python.compat import iterbytes, nativeString, networkString
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_emptyStringLiteral(self):
    """
        Empty string literals are parsed.
        """
    self.server.checker.users = {b'': b''}
    transport = StringTransport()
    self.server.makeConnection(transport)
    transport.clear()
    self.server.dataReceived(b'01 LOGIN {0}\r\n')
    self.assertEqual(transport.value(), b'+ Ready for 0 octets of text\r\n')
    transport.clear()
    self.server.dataReceived(b'{0}\r\n')
    self.assertEqual(transport.value(), b'01 OK LOGIN succeeded\r\n')
    self.assertEqual(self.server.state, 'auth')
    self.server.connectionLost(error.ConnectionDone('Connection done.'))