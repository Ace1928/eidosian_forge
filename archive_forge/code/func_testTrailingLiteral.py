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
def testTrailingLiteral(self):
    transport = StringTransport()
    c = imap4.IMAP4Client()
    c.makeConnection(transport)
    c.lineReceived(b'* OK [IMAP4rev1]')

    def cbCheckTransport(ignored):
        self.assertEqual(transport.value().splitlines()[-1], b'0003 FETCH 1 (RFC822)')

    def cbSelect(ignored):
        d = c.fetchMessage('1')
        c.dataReceived(b'* 1 FETCH (RFC822 {10}\r\n0123456789\r\n RFC822.SIZE 10)\r\n')
        c.dataReceived(b'0003 OK FETCH\r\n')
        d.addCallback(cbCheckTransport)
        return d

    def cbLogin(ignored):
        d = c.select('inbox')
        c.lineReceived(b'0002 OK SELECT')
        d.addCallback(cbSelect)
        return d
    d = c.login(b'blah', b'blah')
    c.dataReceived(b'0001 OK LOGIN\r\n')
    d.addCallback(cbLogin)
    return d