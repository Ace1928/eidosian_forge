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
def assertListDelimiterAndMailboxAreStrings(self, results):
    """
        Assert a C{LIST} response's delimiter and mailbox are native
        strings.

        @param results: A list of tuples as returned by
            L{IMAP4Client.list} or L{IMAP4Client.lsub}.
        """
    for result in results:
        self.assertIsInstance(result[1], str, 'delimiter %r is not a str')
        self.assertIsInstance(result[2], str, 'mailbox %r is not a str')
    return results