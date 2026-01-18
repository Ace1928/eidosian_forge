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
def test_lastOverridesNoneInAdd(self):
    """
        Adding a L{None}, representing C{*}, or a sequence that
        includes L{None} to a L{MessageSet} whose
        L{last<MessageSet.last>} property has been set replaces all
        occurrences of L{None} with the value of
        L{last<MessageSet.last>}.
        """
    hasLast = MessageSet(1)
    hasLast.last = 4
    hasLast.add(None)
    self.assertEqual(list(hasLast), [1, 4])
    self.assertEqual(list(hasLast + (None, 5)), [1, 4, 5])
    hasLast.add(3, None)
    self.assertEqual(list(hasLast), [1, 3, 4])