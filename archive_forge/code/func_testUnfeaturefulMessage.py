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
def testUnfeaturefulMessage(self):
    s = imap4.IMAP4Server()
    f = s._IMAP4Server__cbCopy
    m = FakeMailbox()
    msgs = [FakeyMessage({'Header-Counter': str(i)}, (), b'Date', b'Body %d' % (i,), i + 10, None) for i in range(1, 11)]
    d = f([im for im in zip(range(1, 11), msgs)], 'tag', m)

    def cbCopy(results):
        seen = []
        for a in m.args:
            seen.append(a[0].read())
            self.assertEqual(a[1], ())
            self.assertEqual(a[2], b'Date')
        seen.sort()
        exp = sorted((b'Header-Counter: %d\r\n\r\nBody %d' % (i, i) for i in range(1, 11)))
        self.assertEqual(seen, exp)
        for status, result in results:
            self.assertTrue(status)
            self.assertEqual(result, None)
    return d.addCallback(cbCopy)