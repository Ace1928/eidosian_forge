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
def test_fetchParserBody(self):
    P = imap4._FetchParser
    p = P()
    p.parseString(b'BODY')
    self.assertEqual(len(p.result), 1)
    self.assertTrue(isinstance(p.result[0], p.Body))
    self.assertEqual(p.result[0].peek, False)
    self.assertEqual(p.result[0].header, None)
    self.assertEqual(str(p.result[0]), 'BODY')
    p = P()
    p.parseString(b'BODY.PEEK')
    self.assertEqual(len(p.result), 1)
    self.assertTrue(isinstance(p.result[0], p.Body))
    self.assertEqual(p.result[0].peek, True)
    self.assertEqual(str(p.result[0]), 'BODY')
    p = P()
    p.parseString(b'BODY[]')
    self.assertEqual(len(p.result), 1)
    self.assertTrue(isinstance(p.result[0], p.Body))
    self.assertEqual(p.result[0].empty, True)
    self.assertEqual(str(p.result[0]), 'BODY[]')
    p = P()
    p.parseString(b'BODY[HEADER]')
    self.assertEqual(len(p.result), 1)
    self.assertTrue(isinstance(p.result[0], p.Body))
    self.assertEqual(p.result[0].peek, False)
    self.assertTrue(isinstance(p.result[0].header, p.Header))
    self.assertEqual(p.result[0].header.negate, True)
    self.assertEqual(p.result[0].header.fields, ())
    self.assertEqual(p.result[0].empty, False)
    self.assertEqual(str(p.result[0]), 'BODY[HEADER]')
    p = P()
    p.parseString(b'BODY.PEEK[HEADER]')
    self.assertEqual(len(p.result), 1)
    self.assertTrue(isinstance(p.result[0], p.Body))
    self.assertEqual(p.result[0].peek, True)
    self.assertTrue(isinstance(p.result[0].header, p.Header))
    self.assertEqual(p.result[0].header.negate, True)
    self.assertEqual(p.result[0].header.fields, ())
    self.assertEqual(p.result[0].empty, False)
    self.assertEqual(str(p.result[0]), 'BODY[HEADER]')
    p = P()
    p.parseString(b'BODY[HEADER.FIELDS (Subject Cc Message-Id)]')
    self.assertEqual(len(p.result), 1)
    self.assertTrue(isinstance(p.result[0], p.Body))
    self.assertEqual(p.result[0].peek, False)
    self.assertTrue(isinstance(p.result[0].header, p.Header))
    self.assertEqual(p.result[0].header.negate, False)
    self.assertEqual(p.result[0].header.fields, [b'SUBJECT', b'CC', b'MESSAGE-ID'])
    self.assertEqual(p.result[0].empty, False)
    self.assertEqual(bytes(p.result[0]), b'BODY[HEADER.FIELDS (Subject Cc Message-Id)]')
    p = P()
    p.parseString(b'BODY.PEEK[HEADER.FIELDS (Subject Cc Message-Id)]')
    self.assertEqual(len(p.result), 1)
    self.assertTrue(isinstance(p.result[0], p.Body))
    self.assertEqual(p.result[0].peek, True)
    self.assertTrue(isinstance(p.result[0].header, p.Header))
    self.assertEqual(p.result[0].header.negate, False)
    self.assertEqual(p.result[0].header.fields, [b'SUBJECT', b'CC', b'MESSAGE-ID'])
    self.assertEqual(p.result[0].empty, False)
    self.assertEqual(bytes(p.result[0]), b'BODY[HEADER.FIELDS (Subject Cc Message-Id)]')
    p = P()
    p.parseString(b'BODY.PEEK[HEADER.FIELDS.NOT (Subject Cc Message-Id)]')
    self.assertEqual(len(p.result), 1)
    self.assertTrue(isinstance(p.result[0], p.Body))
    self.assertEqual(p.result[0].peek, True)
    self.assertTrue(isinstance(p.result[0].header, p.Header))
    self.assertEqual(p.result[0].header.negate, True)
    self.assertEqual(p.result[0].header.fields, [b'SUBJECT', b'CC', b'MESSAGE-ID'])
    self.assertEqual(p.result[0].empty, False)
    self.assertEqual(bytes(p.result[0]), b'BODY[HEADER.FIELDS.NOT (Subject Cc Message-Id)]')
    p = P()
    p.parseString(b'BODY[1.MIME]<10.50>')
    self.assertEqual(len(p.result), 1)
    self.assertTrue(isinstance(p.result[0], p.Body))
    self.assertEqual(p.result[0].peek, False)
    self.assertTrue(isinstance(p.result[0].mime, p.MIME))
    self.assertEqual(p.result[0].part, (0,))
    self.assertEqual(p.result[0].partialBegin, 10)
    self.assertEqual(p.result[0].partialLength, 50)
    self.assertEqual(p.result[0].empty, False)
    self.assertEqual(bytes(p.result[0]), b'BODY[1.MIME]<10.50>')
    p = P()
    p.parseString(b'BODY.PEEK[1.3.9.11.HEADER.FIELDS.NOT (Message-Id Date)]<103.69>')
    self.assertEqual(len(p.result), 1)
    self.assertTrue(isinstance(p.result[0], p.Body))
    self.assertEqual(p.result[0].peek, True)
    self.assertTrue(isinstance(p.result[0].header, p.Header))
    self.assertEqual(p.result[0].part, (0, 2, 8, 10))
    self.assertEqual(p.result[0].header.fields, [b'MESSAGE-ID', b'DATE'])
    self.assertEqual(p.result[0].partialBegin, 103)
    self.assertEqual(p.result[0].partialLength, 69)
    self.assertEqual(p.result[0].empty, False)
    self.assertEqual(bytes(p.result[0]), b'BODY[1.3.9.11.HEADER.FIELDS.NOT (Message-Id Date)]<103.69>')