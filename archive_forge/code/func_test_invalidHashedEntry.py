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
def test_invalidHashedEntry(self):
    """
        If there are fewer than three whitespace-separated elements in an
        entry, or the hostname salt/hash portion has more than two elements,
        L{HashedEntry.fromString} should raise L{InvalidEntry}.
        """
    self.invalidEntryTest(HashedEntry)
    a, b, c = sampleHashedLine.split()
    self.assertRaises(InvalidEntry, HashedEntry.fromString, b' '.join([a + b'||', b, c]))