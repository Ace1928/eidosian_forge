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
def test_loginException(self):
    """
        Any exception raised by L{IMAP4Server.authenticateLogin} that
        is not L{UnauthorizedLogin} is logged results in a C{BAD}
        response.
        """

    class UnexpectedException(Exception):
        """
            An unexpected exception.
            """

    def raisesUnexpectedException(user, passwd):
        raise UnexpectedException('Whoops')
    self.server.authenticateLogin = raisesUnexpectedException

    def login():
        return self.client.login(b'testuser', b'password-test')
    d1 = self.connected.addCallback(strip(login))
    d1.addErrback(self.assertClientFailureMessage, b'Server error: Whoops')

    @d1.addCallback
    def assertErrorLogged(_):
        self.assertTrue(self.flushLoggedErrors(UnexpectedException))
    d1.addErrback(self._ebGeneral)
    d1.addBoth(self._cbStopClient)
    d2 = self.loopback()
    d = defer.gatherResults([d1, d2])
    return d.addCallback(self._cbTestFailedLogin)