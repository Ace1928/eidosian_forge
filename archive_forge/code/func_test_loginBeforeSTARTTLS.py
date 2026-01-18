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
def test_loginBeforeSTARTTLS(self):
    """
        A client that attempts to log in before issuing the
        C{STARTTLS} command receives a C{NO} response.
        """
    self.client.startTLS = lambda: defer.succeed(([], 'OK Begin TLS negotiation now'))
    self.connected.addCallback(lambda _: self.client.login(b'wrong', b'time'))
    self.connected.addErrback(self.assertClientFailureMessage, b'LOGIN is disabled before STARTTLS')
    self.connected.addCallback(self._cbStopClient)
    self.connected.addErrback(self._ebGeneral)
    return defer.gatherResults([self.loopback(), self.connected])