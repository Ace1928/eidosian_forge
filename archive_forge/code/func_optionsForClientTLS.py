from __future__ import annotations
import warnings
from binascii import hexlify
from functools import lru_cache
from hashlib import md5
from typing import Dict
from zope.interface import Interface, implementer
from OpenSSL import SSL, crypto
from OpenSSL._util import lib as pyOpenSSLlib
import attr
from constantly import FlagConstant, Flags, NamedConstant, Names
from incremental import Version
from twisted.internet.abstract import isIPAddress, isIPv6Address
from twisted.internet.defer import Deferred
from twisted.internet.error import CertificateError, VerifyError
from twisted.internet.interfaces import (
from twisted.python import log, util
from twisted.python.compat import nativeString
from twisted.python.deprecate import _mutuallyExclusiveArguments, deprecated
from twisted.python.failure import Failure
from twisted.python.randbytes import secureRandom
from ._idna import _idnaBytes
def optionsForClientTLS(hostname, trustRoot=None, clientCertificate=None, acceptableProtocols=None, *, extraCertificateOptions=None):
    """
    Create a L{client connection creator <IOpenSSLClientConnectionCreator>} for
    use with APIs such as L{SSL4ClientEndpoint
    <twisted.internet.endpoints.SSL4ClientEndpoint>}, L{connectSSL
    <twisted.internet.interfaces.IReactorSSL.connectSSL>}, and L{startTLS
    <twisted.internet.interfaces.ITLSTransport.startTLS>}.

    @since: 14.0

    @param hostname: The expected name of the remote host. This serves two
        purposes: first, and most importantly, it verifies that the certificate
        received from the server correctly identifies the specified hostname.
        The second purpose is to use the U{Server Name Indication extension
        <https://en.wikipedia.org/wiki/Server_Name_Indication>} to indicate to
        the server which certificate should be used.
    @type hostname: L{unicode}

    @param trustRoot: Specification of trust requirements of peers. This may be
        a L{Certificate} or the result of L{platformTrust}. By default it is
        L{platformTrust} and you probably shouldn't adjust it unless you really
        know what you're doing. Be aware that clients using this interface
        I{must} verify the server; you cannot explicitly pass L{None} since
        that just means to use L{platformTrust}.
    @type trustRoot: L{IOpenSSLTrustRoot}

    @param clientCertificate: The certificate and private key that the client
        will use to authenticate to the server. If unspecified, the client will
        not authenticate.
    @type clientCertificate: L{PrivateCertificate}

    @param acceptableProtocols: The protocols this peer is willing to speak
        after the TLS negotiation has completed, advertised over both ALPN and
        NPN. If this argument is specified, and no overlap can be found with
        the other peer, the connection will fail to be established. If the
        remote peer does not offer NPN or ALPN, the connection will be
        established, but no protocol wil be negotiated. Protocols earlier in
        the list are preferred over those later in the list.
    @type acceptableProtocols: L{list} of L{bytes}

    @param extraCertificateOptions: A dictionary of additional keyword arguments
        to be presented to L{CertificateOptions}. Please avoid using this unless
        you absolutely need to; any time you need to pass an option here that is
        a bug in this interface.
    @type extraCertificateOptions: L{dict}

    @return: A client connection creator.
    @rtype: L{IOpenSSLClientConnectionCreator}
    """
    if extraCertificateOptions is None:
        extraCertificateOptions = {}
    if trustRoot is None:
        trustRoot = platformTrust()
    if not isinstance(hostname, str):
        raise TypeError('optionsForClientTLS requires text for host names, not ' + hostname.__class__.__name__)
    if clientCertificate:
        extraCertificateOptions.update(privateKey=clientCertificate.privateKey.original, certificate=clientCertificate.original)
    certificateOptions = OpenSSLCertificateOptions(trustRoot=trustRoot, acceptableProtocols=acceptableProtocols, **extraCertificateOptions)
    return ClientTLSOptions(hostname, certificateOptions.getContext())