from __future__ import absolute_import
import hmac
import os
import sys
import warnings
from binascii import hexlify, unhexlify
from hashlib import md5, sha1, sha256
from ..exceptions import (
from ..packages import six
from .url import BRACELESS_IPV6_ADDRZ_RE, IPV4_RE
def assert_fingerprint(cert, fingerprint):
    """
    Checks if given fingerprint matches the supplied certificate.

    :param cert:
        Certificate as bytes object.
    :param fingerprint:
        Fingerprint as string of hexdigits, can be interspersed by colons.
    """
    fingerprint = fingerprint.replace(':', '').lower()
    digest_length = len(fingerprint)
    hashfunc = HASHFUNC_MAP.get(digest_length)
    if not hashfunc:
        raise SSLError('Fingerprint of invalid length: {0}'.format(fingerprint))
    fingerprint_bytes = unhexlify(fingerprint.encode())
    cert_digest = hashfunc(cert).digest()
    if not _const_compare_digest(cert_digest, fingerprint_bytes):
        raise SSLError('Fingerprints did not match. Expected "{0}", got "{1}".'.format(fingerprint, hexlify(cert_digest)))