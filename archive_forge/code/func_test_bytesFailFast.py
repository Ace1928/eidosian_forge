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
def test_bytesFailFast(self):
    """
        If you pass L{bytes} as the hostname to
        L{sslverify.optionsForClientTLS} it immediately raises a L{TypeError}.
        """
    error = self.assertRaises(TypeError, sslverify.optionsForClientTLS, b'not-actually-a-hostname.com')
    expectedText = 'optionsForClientTLS requires text for host names, not ' + bytes.__name__
    self.assertEqual(str(error), expectedText)