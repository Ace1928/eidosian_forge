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
def test_missingPortal(self):
    """
        An L{imap4.IMAP4Server} that is missing a L{Portal} responds
        negatively to an authentication
        """
    self.server.challengers[b'LOGIN'] = imap4.LOGINCredentials
    cAuth = imap4.LOGINAuthenticator(b'testuser')
    self.client.registerAuthenticator(cAuth)
    self.server.portal = None

    def auth():
        return self.client.authenticate(b'secret')
    d = self.connected.addCallback(strip(auth))
    d.addErrback(self.assertClientFailureMessage, b'Temporary authentication failure')
    d.addCallbacks(self._cbStopClient, self._ebGeneral)
    return defer.gatherResults([d, self.loopback()])