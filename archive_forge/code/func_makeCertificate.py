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
def makeCertificate(**kw):
    keypair = PKey()
    keypair.generate_key(TYPE_RSA, 2048)
    certificate = X509()
    certificate.gmtime_adj_notBefore(0)
    certificate.gmtime_adj_notAfter(60 * 60 * 24 * 365)
    for xname in (certificate.get_issuer(), certificate.get_subject()):
        for k, v in kw.items():
            setattr(xname, k, nativeString(v))
    certificate.set_serial_number(counter())
    certificate.set_pubkey(keypair)
    certificate.sign(keypair, 'md5')
    return (keypair, certificate)