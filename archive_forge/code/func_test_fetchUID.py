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
def test_fetchUID(self):
    """
        L{IMAP4Client.fetchUID} sends the I{FETCH UID} command and returns a
        L{Deferred} which fires with a C{dict} mapping message sequence numbers
        to C{dict}s mapping C{'UID'} to that message's I{UID} in the server's
        response.
        """
    d = self.client.fetchUID('1:7')
    self.assertEqual(self.transport.value(), b'0001 FETCH 1:7 (UID)\r\n')
    self.client.lineReceived(b'* 2 FETCH (UID 22)')
    self.client.lineReceived(b'* 3 FETCH (UID 23)')
    self.client.lineReceived(b'* 4 FETCH (UID 24)')
    self.client.lineReceived(b'* 5 FETCH (UID 25)')
    self.client.lineReceived(b'0001 OK FETCH completed')
    self.assertEqual(self.successResultOf(d), {2: {'UID': '22'}, 3: {'UID': '23'}, 4: {'UID': '24'}, 5: {'UID': '25'}})