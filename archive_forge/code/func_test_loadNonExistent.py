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
def test_loadNonExistent(self):
    """
        Loading a L{KnownHostsFile} from a path that does not exist should
        result in an empty L{KnownHostsFile} that will save back to that path.
        """
    pn = self.mktemp()
    knownHostsFile = KnownHostsFile.fromPath(FilePath(pn))
    entries = list(knownHostsFile.iterentries())
    self.assertEqual([], entries)
    self.assertFalse(FilePath(pn).exists())
    knownHostsFile.save()
    self.assertTrue(FilePath(pn).exists())