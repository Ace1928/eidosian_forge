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
def test_ListLineFormatter(self):
    """
        Test that the function which formats the lines in response to a LIST
        command does so appropriately.
        """
    listLines = list(pop3.formatListResponse([]))
    self.assertEqual(listLines, [b'+OK 0\r\n', b'.\r\n'])
    listLines = list(pop3.formatListResponse([1, 2, 3, 100]))
    self.assertEqual(listLines, [b'+OK 4\r\n', b'1 1\r\n', b'2 2\r\n', b'3 3\r\n', b'4 100\r\n', b'.\r\n'])