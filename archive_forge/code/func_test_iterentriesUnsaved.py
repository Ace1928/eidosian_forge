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
def test_iterentriesUnsaved(self):
    """
        If the save path for a L{KnownHostsFile} does not exist,
        L{KnownHostsFile.iterentries} still returns added but unsaved entries.
        """
    hostsFile = KnownHostsFile(FilePath(self.mktemp()))
    hostsFile.addHostKey(b'www.example.com', Key.fromString(sampleKey))
    self.assertEqual(1, len(list(hostsFile.iterentries())))