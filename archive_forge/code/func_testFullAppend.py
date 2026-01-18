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
def testFullAppend(self):
    infile = util.sibpath(__file__, 'rfc822.message')
    SimpleServer.theAccount.addMailbox('root/subthing')

    def login():
        return self.client.login(b'testuser', b'password-test')

    @defer.inlineCallbacks
    def append():
        with open(infile, 'rb') as message:
            result = (yield self.client.append('root/subthing', message, ('\\SEEN', '\\DELETED'), 'Tue, 17 Jun 2003 11:22:16 -0600 (MDT)'))
            defer.returnValue(result)
    d1 = self.connected.addCallback(strip(login))
    d1.addCallbacks(strip(append), self._ebGeneral)
    d1.addCallbacks(self._cbStopClient, self._ebGeneral)
    d2 = self.loopback()
    d = defer.gatherResults([d1, d2])
    return d.addCallback(self._cbTestFullAppend, infile)