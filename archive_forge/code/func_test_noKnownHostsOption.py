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
def test_noKnownHostsOption(self):
    """
        L{default.verifyHostKey} should find your known_hosts file in
        ~/.ssh/known_hosts if you don't specify one explicitly on the command
        line.
        """
    l = []
    tmpdir = self.mktemp()
    oldHostsOption = self.hostsOption
    hostsNonOption = FilePath(tmpdir).child('.ssh').child('known_hosts')
    hostsNonOption.parent().makedirs()
    FilePath(oldHostsOption).moveTo(hostsNonOption)
    self.replaceHome(tmpdir)
    self.options['known-hosts'] = None
    default.verifyHostKey(self.fakeTransport, b'4.3.2.1', sampleKey, b"I don't care.").addCallback(l.append)
    self.assertEqual([1], l)