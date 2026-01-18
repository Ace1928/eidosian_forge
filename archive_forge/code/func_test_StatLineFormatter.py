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
def test_StatLineFormatter(self):
    """
        Test that the function which formats stat lines does so appropriately.
        """
    statLine = list(pop3.formatStatResponse([]))[-1]
    self.assertEqual(statLine, b'+OK 0 0\r\n')
    statLine = list(pop3.formatStatResponse([10, 31, 0, 10101]))[-1]
    self.assertEqual(statLine, b'+OK 4 10142\r\n')