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
def test_extraChainFilesAreAddedIfSupplied(self):
    """
        If C{extraCertChain} is set and all prerequisites are met, the
        specified chain certificates are added to C{Context}s that get
        created.
        """
    opts = sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, extraCertChain=self.extraCertChain)
    opts._contextFactory = FakeContext
    ctx = opts.getContext()
    self.assertEqual(self.sKey, ctx._privateKey)
    self.assertEqual(self.sCert, ctx._certificate)
    self.assertEqual(self.extraCertChain, ctx._extraCertChain)