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
def test_get_elliptic_curveReturns(self):
    """
        L{FakeCrypto.test_get_elliptic_curve} returns the value
        specified by its state object and records what it was called
        with.
        """
    returnValue = 'object'
    state = FakeCryptoState(getEllipticCurveRaises=None, getEllipticCurveReturns=returnValue)
    crypto = FakeCrypto(state)
    self.assertIs(crypto.get_elliptic_curve('another curve name'), returnValue)
    self.assertEqual(state.getEllipticCurveCalls, ['another curve name'])