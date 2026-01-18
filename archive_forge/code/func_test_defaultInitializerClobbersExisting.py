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
def test_defaultInitializerClobbersExisting(self):
    """
        After using the default initializer for L{KnownHostsFile}, the first use
        of L{KnownHostsFile.save} overwrites any existing contents in the save
        path.
        """
    path = self.pathWithContent(sampleHashedLine)
    hostsFile = KnownHostsFile(path)
    entry = hostsFile.addHostKey(b'www.example.com', Key.fromString(otherSampleKey))
    hostsFile.save()
    self.assertEqual([entry], list(hostsFile.iterentries()))
    self.assertEqual(entry.toString() + b'\n', path.getContent())