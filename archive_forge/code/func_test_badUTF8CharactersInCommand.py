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
def test_badUTF8CharactersInCommand(self):
    """
        Sending a command with invalid UTF-8 characters
        will raise a L{pop3.POP3Error}.
        """
    error = b'not authenticated yet: cannot do \x81PASS'
    d = self.runTest([b'\x81PASS', b'QUIT'], [b'+OK <moshez>', b'-ERR bad protocol or server: POP3Error: ' + error, b'+OK '])
    errors = self.flushLoggedErrors(pop3.POP3Error)
    self.assertEqual(len(errors), 1)
    return d