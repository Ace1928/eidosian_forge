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
def flushPending(self, asLongAs=lambda: True):
    """
        Advance pending iterators enqueued with L{iterateInReactor} in
        a round-robin fashion, resuming the transport's producer until
        it has completed.  This ensures bodies are flushed.

        @param asLongAs: (optional) An optional predicate function.
            Flushing iterators continues as long as there are
            iterators and this returns L{True}.
        """
    while self.iterators and asLongAs():
        for e in self.iterators[0][0]:
            while self.transport.producer:
                self.transport.producer.resumeProducing()
        else:
            self.iterators.pop(0)[1].callback(None)