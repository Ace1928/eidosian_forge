import os
import hamcrest
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet.error import ConnectionDone
from twisted.internet.testing import waitUntilAllDisconnected
from twisted.protocols import basic
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.test.test_tcp import ProperlyCloseFilesMixin
from twisted.trial.unittest import TestCase
from zope.interface import implementer
def makeContextFactory(self, org, orgUnit, *args, **kwArgs):
    base = self.mktemp()
    generateCertificateFiles(base, org, orgUnit)
    serverCtxFactory = ssl.DefaultOpenSSLContextFactory(os.extsep.join((base, 'key')), os.extsep.join((base, 'cert')), *args, **kwArgs)
    return (base, serverCtxFactory)