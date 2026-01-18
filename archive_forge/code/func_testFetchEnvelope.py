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
def testFetchEnvelope(self, uid=0):
    self.function = self.client.fetchEnvelope
    self.messages = '15'
    self.msgObjs = [FakeyMessage({'from': 'user@domain', 'to': 'resu@domain', 'date': 'thursday', 'subject': 'it is a message', 'message-id': 'id-id-id-yayaya'}, (), b'', b'', 65656, None)]
    self.expected = {0: {'ENVELOPE': ['thursday', 'it is a message', [[None, None, 'user', 'domain']], [[None, None, 'user', 'domain']], [[None, None, 'user', 'domain']], [[None, None, 'resu', 'domain']], None, None, None, 'id-id-id-yayaya']}}
    return self._fetchWork(uid)