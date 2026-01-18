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
def test_butIfTheyDidItWouldWork(self):
    """
        L{ssl.optionsForClientTLS} should be using L{ssl.platformTrust} by
        default, so if we fake that out then it should trust ourselves again.
        """
    cProto, sProto, cWrapped, sWrapped, pump = self.serviceIdentitySetup('valid.example.com', 'valid.example.com', useDefaultTrust=True, fakePlatformTrust=True)
    self.assertEqual(cWrapped.data, b'greetings!')
    cErr = cWrapped.lostReason
    sErr = sWrapped.lostReason
    self.assertIsNone(cErr)
    self.assertIsNone(sErr)