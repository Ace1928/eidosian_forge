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
def test_equalityIterationAndAddition(self):
    """
        Test the following properties of L{MessageSet} addition and
        equality:

            1. Two empty L{MessageSet}s are equal to each other;

            2. A L{MessageSet} is not equal to any other object;

            2. Adding a L{MessageSet} and another L{MessageSet} or an
               L{int} representing a single message or a sequence of
               L{int}s representing a sequence of message numbers
               produces a new L{MessageSet} that:

            3. Has a length equal to the number of messages within
               each sequence of message numbers;

            4. Yields each message number in ascending order when
               iterated over;

            6. L{MessageSet.add} with a single message or a start and
               end message satisfies 3 and 4 above.
        """
    m1 = MessageSet()
    m2 = MessageSet()
    self.assertEqual(m1, m2)
    self.assertNotEqual(m1, ())
    m1 = m1 + 1
    self.assertEqual(len(m1), 1)
    self.assertEqual(list(m1), [1])
    m1 = m1 + (1, 3)
    self.assertEqual(len(m1), 3)
    self.assertEqual(list(m1), [1, 2, 3])
    m2 = m2 + (1, 3)
    self.assertEqual(m1, m2)
    self.assertEqual(list(m1 + m2), [1, 2, 3])
    m1.add(5)
    self.assertEqual(len(m1), 4)
    self.assertEqual(list(m1), [1, 2, 3, 5])
    self.assertNotEqual(m1, m2)
    m1.add(6, 8)
    self.assertEqual(len(m1), 7)
    self.assertEqual(list(m1), [1, 2, 3, 5, 6, 7, 8])