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
def testInspectDistinguishedName(self):
    n = sslverify.DN(commonName=b'common name', organizationName=b'organization name', organizationalUnitName=b'organizational unit name', localityName=b'locality name', stateOrProvinceName=b'state or province name', countryName=b'country name', emailAddress=b'email address')
    s = n.inspect()
    for k in ['common name', 'organization name', 'organizational unit name', 'locality name', 'state or province name', 'country name', 'email address']:
        self.assertIn(k, s, f'{k!r} was not in inspect output.')
        self.assertIn(k.title(), s, f'{k!r} was not in inspect output.')