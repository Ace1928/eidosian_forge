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
def test_STAT(self):
    """
        Test the single form of the STAT command, which returns a short-form
        response of the number of messages in the mailbox and their total size.
        """
    p = self.pop3Server
    s = self.pop3Transport
    p.lineReceived(b'STAT')
    self._flush()
    self.assertEqual(s.getvalue(), b'+OK 1 44\r\n')