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
def testInspectDistinguishedNameWithoutAllFields(self):
    n = sslverify.DN(localityName=b'locality name')
    s = n.inspect()
    for k in ['common name', 'organization name', 'organizational unit name', 'state or province name', 'country name', 'email address']:
        self.assertNotIn(k, s, f'{k!r} was in inspect output.')
        self.assertNotIn(k.title(), s, f'{k!r} was in inspect output.')
    self.assertIn('locality name', s)
    self.assertIn('Locality Name', s)