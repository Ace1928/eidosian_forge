import datetime
import itertools
import sys
from unittest import skipIf
from zope.interface import implementer
from incremental import Version
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet._idna import _idnaText
from twisted.internet.error import CertificateError, ConnectionClosed, ConnectionLost
from twisted.internet.task import Clock
from twisted.python.compat import nativeString
from twisted.python.filepath import FilePath
from twisted.python.modules import getModule
from twisted.python.reflect import requireModule
from twisted.test.iosim import connectedServerAndClient
from twisted.test.test_twisted import SetAsideModule
from twisted.trial import util
from twisted.trial.unittest import SkipTest, SynchronousTestCase, TestCase
def test_dhParams(self):
    """
        If C{dhParams} is set, they are loaded into each new context.
        """

    class FakeDiffieHellmanParameters:
        _dhFile = FilePath(b'dh.params')
    dhParams = FakeDiffieHellmanParameters()
    opts = sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, dhParameters=dhParams)
    opts._contextFactory = FakeContext
    ctx = opts.getContext()
    self.assertEqual(FakeDiffieHellmanParameters._dhFile.path, ctx._dhFilename)