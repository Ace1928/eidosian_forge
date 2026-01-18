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
def test_flagsChangedInsideFetchSpecificResponse(self):
    """
        Any unrequested flag information received along with other requested
        information in an untagged I{FETCH} received in response to a request
        issued with L{IMAP4Client.fetchSpecific} is passed to the
        C{flagsChanged} callback.
        """
    transport = StringTransport()
    c = StillSimplerClient()
    c.makeConnection(transport)
    c.lineReceived(b'* OK [IMAP4rev1]')

    def login():
        d = c.login(b'blah', b'blah')
        c.dataReceived(b'0001 OK LOGIN\r\n')
        return d

    def select():
        d = c.select('inbox')
        c.lineReceived(b'0002 OK SELECT')
        return d

    def fetch():
        d = c.fetchSpecific(b'1:*', headerType='HEADER.FIELDS', headerArgs=['SUBJECT'])
        c.dataReceived(b'* 1 FETCH (BODY[HEADER.FIELDS ("SUBJECT")] {22}\r\n')
        c.dataReceived(b'Subject: subject one\r\n')
        c.dataReceived(b' FLAGS (\\Recent))\r\n')
        c.dataReceived(b'* 2 FETCH (FLAGS (\\Seen) BODY[HEADER.FIELDS ("SUBJECT")] {22}\r\n')
        c.dataReceived(b'Subject: subject two\r\n')
        c.dataReceived(b')\r\n')
        c.dataReceived(b'0003 OK FETCH completed\r\n')
        return d

    def test(res):
        self.assertEqual(res, {1: [['BODY', ['HEADER.FIELDS', ['SUBJECT']], 'Subject: subject one\r\n']], 2: [['BODY', ['HEADER.FIELDS', ['SUBJECT']], 'Subject: subject two\r\n']]})
        self.assertEqual(c.flags, {1: ['\\Recent'], 2: ['\\Seen']})
    return login().addCallback(strip(select)).addCallback(strip(fetch)).addCallback(test)