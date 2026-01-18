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
def test_hashedNotBase64(self):
    """
        If the key, host salt, or host hash portion of a hashed entry is not
        encoded, it will raise L{BinasciiError}.
        """
    self.notBase64Test(HashedEntry)
    a, b, c = sampleHashedLine.split()
    self.assertRaises(BinasciiError, HashedEntry.fromString, b' '.join([b'|1|x|' + b2a_base64(b'stuff').strip(), b, c]))
    self.assertRaises(BinasciiError, HashedEntry.fromString, b' '.join([HashedEntry.MAGIC + b2a_base64(b'stuff').strip() + b'|x', b, c]))
    self.assertRaises(BinasciiError, HashedEntry.fromString, b' '.join([b'|1|x|x', b, c]))