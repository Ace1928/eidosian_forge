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
def testUndecodableName(self):
    """
        C{STATUS} names that cannot be decoded as ASCII cause the
        status Deferred to fail with L{IllegalServerResponse}
        """
    d = self.client.status('blurdybloop', 'MESSAGES')
    self.assertEqual(self.transport.value(), b'0001 STATUS blurdybloop (MESSAGES)\r\n')
    self.client.lineReceived(b'* STATUS blurdybloop (MESSAGES 1 ASCIINAME "OK" NOT\xffASCII "NO")')
    self.client.lineReceived(b'0001 OK STATUS completed')
    self.failureResultOf(d, imap4.IllegalServerResponse)