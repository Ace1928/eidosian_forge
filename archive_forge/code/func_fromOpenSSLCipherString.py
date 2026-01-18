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
@classmethod
def fromOpenSSLCipherString(cls, cipherString):
    """
        Create a new instance using an OpenSSL cipher string.

        @param cipherString: An OpenSSL cipher string that describes what
            cipher suites are acceptable.
            See the documentation of U{OpenSSL
            <http://www.openssl.org/docs/apps/ciphers.html#CIPHER_STRINGS>} or
            U{Apache
            <http://httpd.apache.org/docs/2.4/mod/mod_ssl.html#sslciphersuite>}
            for details.
        @type cipherString: L{unicode}

        @return: Instance representing C{cipherString}.
        @rtype: L{twisted.internet.ssl.AcceptableCiphers}
        """
    return cls(_expandCipherString(nativeString(cipherString), SSL.TLS_METHOD, SSL.OP_NO_SSLv2 | SSL.OP_NO_SSLv3))