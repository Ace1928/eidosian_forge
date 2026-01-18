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
def test_hasLaterAddedKey(self):
    """
        L{KnownHostsFile.hasHostKey} returns C{True} when a key for the given
        hostname is present in the file, even if it is only added to the file
        after the L{KnownHostsFile} instance is initialized.
        """
    key = Key.fromString(sampleKey)
    entry = PlainEntry([b'brandnew.example.com'], key.sshType(), key, b'')
    hostsFile = self.loadSampleHostsFile()
    with hostsFile.savePath.open('a') as hostsFileObj:
        hostsFileObj.write(entry.toString() + b'\n')
    self.assertEqual(True, hostsFile.hasHostKey(b'brandnew.example.com', key))