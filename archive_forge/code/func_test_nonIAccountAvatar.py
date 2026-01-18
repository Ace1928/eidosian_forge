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
def test_nonIAccountAvatar(self):
    """
        The server responds with a C{BAD} response when its portal
        attempts to log a user in with checker that claims to support
        L{IAccount} but returns an an avatar interface that is not
        L{IAccount}.
        """

    def brokenRequestAvatar(*_, **__):
        return ('Not IAccount', 'Not an account', lambda: None)
    self.server.portal.realm.requestAvatar = brokenRequestAvatar

    def login():
        d = self.client.login(b'testuser', b'password-test')
        d.addBoth(self._cbStopClient)
    d1 = self.connected.addCallback(strip(login)).addErrback(self._ebGeneral)
    d2 = self.loopback()
    d = defer.gatherResults([d1, d2])
    return d.addCallback(self._cbTestFailedLogin)