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
def loopbackTLSConnectionInMemory(trustRoot, privateKey, serverCertificate, clientProtocols=None, serverProtocols=None, clientOptions=None):
    """
    Create a loopback TLS connection with the given trust and keys. Like
    L{loopbackTLSConnection}, but using in-memory certificates and keys rather
    than writing them to disk.

    @param trustRoot: the C{trustRoot} argument for the client connection's
        context.
    @type trustRoot: L{sslverify.IOpenSSLTrustRoot}

    @param privateKey: The private key.
    @type privateKey: L{str} (native string)

    @param serverCertificate: The certificate used by the server.
    @type chainedCertFile: L{str} (native string)

    @param clientProtocols: The protocols the client is willing to negotiate
        using NPN/ALPN.

    @param serverProtocols: The protocols the server is willing to negotiate
        using NPN/ALPN.

    @param clientOptions: The type of C{OpenSSLCertificateOptions} class to
        use for the client. Defaults to C{OpenSSLCertificateOptions}.

    @return: 3-tuple of server-protocol, client-protocol, and L{IOPump}
    @rtype: L{tuple}
    """
    if clientOptions is None:
        clientOptions = sslverify.OpenSSLCertificateOptions
    clientCertOpts = clientOptions(trustRoot=trustRoot, acceptableProtocols=clientProtocols)
    serverCertOpts = sslverify.OpenSSLCertificateOptions(privateKey=privateKey, certificate=serverCertificate, acceptableProtocols=serverProtocols)
    return _loopbackTLSConnection(serverCertOpts, clientCertOpts)