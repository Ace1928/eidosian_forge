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
def patchedOpen(self, fname, mode, **kwargs):
    """
        The patched version of 'open'; this returns a L{FakeFile} that the
        instantiated L{ConsoleUI} can use.
        """
    self.assertEqual(fname, '/dev/tty')
    self.assertEqual(mode, 'r+b')
    self.assertEqual(kwargs['buffering'], 0)
    return self.fakeFile