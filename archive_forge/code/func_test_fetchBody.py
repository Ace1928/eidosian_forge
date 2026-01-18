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
def test_fetchBody(self):
    """
        L{IMAP4Client.fetchBody} sends the I{FETCH BODY} command and returns a
        L{Deferred} which fires with a C{dict} mapping message sequence numbers
        to C{dict}s mapping C{'RFC822.TEXT'} to that message's body as given in
        the server's response.
        """
    d = self.client.fetchBody('3')
    self.assertEqual(self.transport.value(), b'0001 FETCH 3 (RFC822.TEXT)\r\n')
    self.client.lineReceived(b'* 3 FETCH (RFC822.TEXT "Message text")')
    self.client.lineReceived(b'0001 OK FETCH completed')
    self.assertEqual(self.successResultOf(d), {3: {'RFC822.TEXT': 'Message text'}})