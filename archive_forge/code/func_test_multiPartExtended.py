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
def test_multiPartExtended(self):
    """
        When passed a I{multipart/*} message and C{True} for the C{extended}
        argument, L{imap4.getBodyStructure} includes extended structure
        information from the parts of the multipart message and extended
        structure information about the multipart message itself.
        """
    oneSubPart = FakeyMessage({b'content-type': b'image/jpeg; x=y', b'content-id': b'some kind of id', b'content-description': b'great justice', b'content-transfer-encoding': b'maximum'}, (), b'', b'hello world', 123, None)
    anotherSubPart = FakeyMessage({b'content-type': b'text/plain; charset=us-ascii'}, (), b'', b'some stuff', 321, None)
    container = FakeyMessage({'content-type': 'multipart/related; foo=bar', 'content-language': 'es', 'content-location': 'Spain', 'content-disposition': 'attachment; name=monkeys'}, (), b'', b'', 555, [oneSubPart, anotherSubPart])
    self.assertEqual([imap4.getBodyStructure(oneSubPart, extended=True), imap4.getBodyStructure(anotherSubPart, extended=True), 'related', ['foo', 'bar'], ['attachment', ['name', 'monkeys']], 'es', 'Spain'], imap4.getBodyStructure(container, extended=True))