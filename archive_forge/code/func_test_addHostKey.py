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
def test_addHostKey(self):
    """
        L{KnownHostsFile.addHostKey} adds a new L{HashedEntry} to the host
        file, and returns it.
        """
    hostsFile = self.loadSampleHostsFile()
    aKey = Key.fromString(thirdSampleKey)
    self.assertEqual(False, hostsFile.hasHostKey(b'somewhere.example.com', aKey))
    newEntry = hostsFile.addHostKey(b'somewhere.example.com', aKey)
    self.assertEqual(20, len(newEntry._hostSalt))
    self.assertEqual(True, newEntry.matchesHost(b'somewhere.example.com'))
    self.assertEqual(newEntry.keyType, b'ssh-rsa')
    self.assertEqual(aKey, newEntry.publicKey)
    self.assertEqual(True, hostsFile.hasHostKey(b'somewhere.example.com', aKey))