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
def signRequestObject(self, issuerDistinguishedName, requestObject, serialNumber, secondsToExpiry=60 * 60 * 24 * 365, digestAlgorithm='sha256'):
    """
        Sign a CertificateRequest instance, returning a Certificate instance.
        """
    req = requestObject.original
    cert = crypto.X509()
    issuerDistinguishedName._copyInto(cert.get_issuer())
    cert.set_subject(req.get_subject())
    cert.set_pubkey(req.get_pubkey())
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(secondsToExpiry)
    cert.set_serial_number(serialNumber)
    cert.sign(self.original, digestAlgorithm)
    return Certificate(cert)