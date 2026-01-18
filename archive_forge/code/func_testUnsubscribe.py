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
def testUnsubscribe(self):
    SimpleServer.theAccount.subscriptions = ['THIS/MBOX', 'THAT/MBOX']

    def login():
        return self.client.login(b'testuser', b'password-test')

    def unsubscribe():
        return self.client.unsubscribe('this/mbox')
    d1 = self.connected.addCallback(strip(login))
    d1.addCallbacks(strip(unsubscribe), self._ebGeneral)
    d1.addCallbacks(self._cbStopClient, self._ebGeneral)
    d2 = self.loopback()
    d = defer.gatherResults([d1, d2])
    d.addCallback(lambda _: self.assertEqual(SimpleServer.theAccount.subscriptions, ['THAT/MBOX']))
    return d