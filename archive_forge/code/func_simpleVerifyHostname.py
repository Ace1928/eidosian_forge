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
def simpleVerifyHostname(connection, hostname):
    """
    Check only the common name in the certificate presented by the peer and
    only for an exact match.

    This is to provide I{something} in the way of hostname verification to
    users who haven't installed C{service_identity}. This check is overly
    strict, relies on a deprecated TLS feature (you're supposed to ignore the
    commonName if the subjectAlternativeName extensions are present, I
    believe), and lots of valid certificates will fail.

    @param connection: the OpenSSL connection to verify.
    @type connection: L{OpenSSL.SSL.Connection}

    @param hostname: The hostname expected by the user.
    @type hostname: L{unicode}

    @raise twisted.internet.ssl.VerificationError: if the common name and
        hostname don't match.
    """
    commonName = connection.get_peer_certificate().get_subject().commonName
    if commonName != hostname:
        raise SimpleVerificationError(repr(commonName) + '!=' + repr(hostname))