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
def test_multiPart(self):
    """
        For a I{multipart/*} message, L{imap4.getBodyStructure} returns a list
        containing the body structure information for each part of the message
        followed by an element giving the MIME subtype of the message.
        """
    oneSubPart = FakeyMessage({'content-type': 'image/jpeg; x=y', 'content-id': 'some kind of id', 'content-description': 'great justice', 'content-transfer-encoding': 'maximum'}, (), b'', b'hello world', 123, None)
    anotherSubPart = FakeyMessage({'content-type': 'text/plain; charset=us-ascii'}, (), b'', b'some stuff', 321, None)
    container = FakeyMessage({'content-type': 'multipart/related'}, (), b'', b'', 555, [oneSubPart, anotherSubPart])
    self.assertEqual([imap4.getBodyStructure(oneSubPart), imap4.getBodyStructure(anotherSubPart), 'related'], imap4.getBodyStructure(container))