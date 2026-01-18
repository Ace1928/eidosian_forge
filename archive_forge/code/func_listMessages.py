import base64
import hmac
import itertools
from collections import OrderedDict
from hashlib import md5
from io import BytesIO
from zope.interface import implementer
from zope.interface.verify import verifyClass
import twisted.cred.checkers
import twisted.cred.portal
import twisted.internet.protocol
import twisted.mail.pop3
import twisted.mail.protocols
from twisted import cred, internet, mail
from twisted.cred.credentials import IUsernameHashedPassword
from twisted.internet import defer
from twisted.internet.testing import LineSendingProtocol
from twisted.mail import pop3
from twisted.protocols import loopback
from twisted.python import failure
from twisted.trial import unittest, util
def listMessages(self, n=None):
    """
        Record a new unfired L{Deferred} in C{self.waiting} and return it.

        @type n: L{int} or L{None}
        @param n: The 0-based index of the message.

        @return: The L{Deferred}
        """
    d = defer.Deferred()
    self.waiting.append((d, DummyMailbox.listMessages(self, n)))
    return d