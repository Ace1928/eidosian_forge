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
def negotiateProtocol(serverProtocols, clientProtocols, clientOptions=None):
    """
    Create the TLS connection and negotiate a next protocol.

    @param serverProtocols: The protocols the server is willing to negotiate.
    @param clientProtocols: The protocols the client is willing to negotiate.
    @param clientOptions: The type of C{OpenSSLCertificateOptions} class to
        use for the client. Defaults to C{OpenSSLCertificateOptions}.
    @return: A L{tuple} of the negotiated protocol and the reason the
        connection was lost.
    """
    caCertificate, serverCertificate = certificatesForAuthorityAndServer()
    trustRoot = sslverify.OpenSSLCertificateAuthorities([caCertificate.original])
    sProto, cProto, sWrapped, cWrapped, pump = loopbackTLSConnectionInMemory(trustRoot=trustRoot, privateKey=serverCertificate.privateKey.original, serverCertificate=serverCertificate.original, clientProtocols=clientProtocols, serverProtocols=serverProtocols, clientOptions=clientOptions)
    pump.flush()
    return (cProto.negotiatedProtocol, cWrapped.lostReason)