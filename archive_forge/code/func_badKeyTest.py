import os
from binascii import Error as BinasciiError, a2b_base64, b2a_base64
from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.conch.error import HostKeyChanged, InvalidEntry, UserRejectedKey
from twisted.conch.interfaces import IKnownHostEntry
from twisted.internet.defer import Deferred
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.test.testutils import ComparisonTestsMixin
from twisted.trial.unittest import TestCase
def badKeyTest(self, cls, prefix):
    """
        If the key portion of the entry is valid base64, but is not actually an
        SSH key, C{fromString} should raise L{BadKeyError}.
        """
    self.assertRaises(BadKeyError, cls.fromString, b' '.join([prefix, b'ssh-rsa', b2a_base64(b"Hey, this isn't an SSH key!").strip()]))