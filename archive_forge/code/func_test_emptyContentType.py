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
def test_emptyContentType(self):
    """
        L{imap4.getBodyStructure} returns L{None} for the major and
        minor MIME types of a L{IMessagePart} provider whose headers
        lack a C{Content-Type}, or have an empty value for it.
        """
    missing = FakeyMessage({}, (), b'', b'', 123, None)
    missingContentTypeStructure = imap4.getBodyStructure(missing)
    missingMajor, missingMinor = missingContentTypeStructure[:2]
    self.assertIs(None, missingMajor)
    self.assertIs(None, missingMinor)
    empty = FakeyMessage({'content-type': ''}, (), b'', b'', 123, None)
    emptyContentTypeStructure = imap4.getBodyStructure(empty)
    emptyMajor, emptyMinor = emptyContentTypeStructure[:2]
    self.assertIs(None, emptyMajor)
    self.assertIs(None, emptyMinor)
    newline = FakeyMessage({'content-type': '\n'}, (), b'', b'', 123, None)
    newlineContentTypeStructure = imap4.getBodyStructure(newline)
    newlineMajor, newlineMinor = newlineContentTypeStructure[:2]
    self.assertIs(None, newlineMajor)
    self.assertIs(None, newlineMinor)