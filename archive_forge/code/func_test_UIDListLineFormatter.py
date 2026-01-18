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
def test_UIDListLineFormatter(self):
    """
        Test that the function which formats lines in response to a UIDL
        command does so appropriately.
        """
    uids = ['abc', 'def', 'ghi']
    listLines = list(pop3.formatUIDListResponse([], uids.__getitem__))
    self.assertEqual(listLines, [b'+OK \r\n', b'.\r\n'])
    listLines = list(pop3.formatUIDListResponse([123, 431, 591], uids.__getitem__))
    self.assertEqual(listLines, [b'+OK \r\n', b'1 abc\r\n', b'2 def\r\n', b'3 ghi\r\n', b'.\r\n'])
    listLines = list(pop3.formatUIDListResponse([0, None, 591], uids.__getitem__))
    self.assertEqual(listLines, [b'+OK \r\n', b'1 abc\r\n', b'3 ghi\r\n', b'.\r\n'])