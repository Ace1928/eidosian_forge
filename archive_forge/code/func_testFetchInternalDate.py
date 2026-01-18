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
def testFetchInternalDate(self, uid=0):
    self.function = self.client.fetchInternalDate
    self.messages = '13'
    self.msgObjs = [FakeyMessage({}, (), b'Fri, 02 Nov 2003 21:25:10 GMT', b'', 23232, None), FakeyMessage({}, (), b'Thu, 29 Dec 2013 11:31:52 EST', b'', 101, None), FakeyMessage({}, (), b'Mon, 10 Mar 1992 02:44:30 CST', b'', 202, None), FakeyMessage({}, (), b'Sat, 11 Jan 2000 14:40:24 PST', b'', 303, None)]
    self.expected = {0: {'INTERNALDATE': '02-Nov-2003 21:25:10 +0000'}, 1: {'INTERNALDATE': '29-Dec-2013 11:31:52 -0500'}, 2: {'INTERNALDATE': '10-Mar-1992 02:44:30 -0600'}, 3: {'INTERNALDATE': '11-Jan-2000 14:40:24 -0800'}}
    return self._fetchWork(uid)