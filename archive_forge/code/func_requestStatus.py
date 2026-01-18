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
def requestStatus(self, names):
    """
        Return the mailbox's status.

        @param names: The status items to include.

        @return: A L{dict} of status data.
        """
    r = {}
    if 'MESSAGES' in names:
        r['MESSAGES'] = self.getMessageCount()
    if 'RECENT' in names:
        r['RECENT'] = self.getRecentCount()
    if 'UIDNEXT' in names:
        r['UIDNEXT'] = self.getMessageCount() + 1
    if 'UIDVALIDITY' in names:
        r['UIDVALIDITY'] = self.getUID()
    if 'UNSEEN' in names:
        r['UNSEEN'] = self.getUnseenCount()
    return defer.succeed(r)