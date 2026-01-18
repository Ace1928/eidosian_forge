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
def test_literalWithoutPrecedingWhitespace(self):
    """
        Literals should be recognized even when they are not preceded by
        whitespace.
        """
    transport = StringTransport()
    protocol = imap4.IMAP4Client()
    protocol.makeConnection(transport)
    protocol.lineReceived(b'* OK [IMAP4rev1]')

    def login():
        d = protocol.login(b'blah', b'blah')
        protocol.dataReceived(b'0001 OK LOGIN\r\n')
        return d

    def select():
        d = protocol.select(b'inbox')
        protocol.lineReceived(b'0002 OK SELECT')
        return d

    def fetch():
        d = protocol.fetchSpecific('1:*', headerType='HEADER.FIELDS', headerArgs=['SUBJECT'])
        protocol.dataReceived(b'* 1 FETCH (BODY[HEADER.FIELDS ({7}\r\nSUBJECT)] "Hello")\r\n')
        protocol.dataReceived(b'0003 OK FETCH completed\r\n')
        return d

    def test(result):
        self.assertEqual(transport.value().splitlines()[-1], b'0003 FETCH 1:* BODY[HEADER.FIELDS (SUBJECT)]')
        self.assertEqual(result, {1: [['BODY', ['HEADER.FIELDS', ['SUBJECT']], 'Hello']]})
    d = login()
    d.addCallback(strip(select))
    d.addCallback(strip(fetch))
    d.addCallback(test)
    return d