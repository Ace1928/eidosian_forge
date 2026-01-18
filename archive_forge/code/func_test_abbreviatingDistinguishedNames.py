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
def test_abbreviatingDistinguishedNames(self):
    """
        Check that abbreviations used in certificates correctly map to
        complete names.
        """
    self.assertEqual(sslverify.DN(CN=b'a', OU=b'hello'), sslverify.DistinguishedName(commonName=b'a', organizationalUnitName=b'hello'))
    self.assertNotEqual(sslverify.DN(CN=b'a', OU=b'hello'), sslverify.DN(CN=b'a', OU=b'hello', emailAddress=b'xxx'))
    dn = sslverify.DN(CN=b'abcdefg')
    self.assertRaises(AttributeError, setattr, dn, 'Cn', b'x')
    self.assertEqual(dn.CN, dn.commonName)
    dn.CN = b'bcdefga'
    self.assertEqual(dn.CN, dn.commonName)