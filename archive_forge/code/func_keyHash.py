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
def keyHash(self):
    """
        Compute a hash of the underlying PKey object.

        The purpose of this method is to allow you to determine if two
        certificates share the same public key; it is not really useful for
        anything else.

        In versions of Twisted prior to 15.0, C{keyHash} used a technique
        involving certificate requests for computing the hash that was not
        stable in the face of changes to the underlying OpenSSL library.

        @return: Return a 32-character hexadecimal string uniquely identifying
            this public key, I{for this version of Twisted}.
        @rtype: native L{str}
        """
    raw = crypto.dump_publickey(crypto.FILETYPE_ASN1, self.original)
    h = md5()
    h.update(raw)
    return h.hexdigest()