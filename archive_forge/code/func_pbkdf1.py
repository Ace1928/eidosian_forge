from __future__ import division
import hashlib
import logging; log = logging.getLogger(__name__)
import re
import os
from struct import Struct
from warnings import warn
from passlib import exc
from passlib.utils import join_bytes, to_native_str, join_byte_values, to_bytes, \
from passlib.utils.compat import irange, int_types, unicode_or_bytes_types, PY3, error_from
from passlib.utils.decor import memoized_property
def pbkdf1(digest, secret, salt, rounds, keylen=None):
    """pkcs#5 password-based key derivation v1.5

    :arg digest:
        digest name or constructor.
        
    :arg secret:
        secret to use when generating the key.
        may be :class:`!bytes` or :class:`unicode` (encoded using UTF-8).
        
    :arg salt:
        salt string to use when generating key.
        may be :class:`!bytes` or :class:`unicode` (encoded using UTF-8).

    :param rounds:
        number of rounds to use to generate key.

    :arg keylen:
        number of bytes to generate (if omitted / ``None``, uses digest's native size)

    :returns:
        raw :class:`bytes` of generated key

    .. note::

        This algorithm has been deprecated, new code should use PBKDF2.
        Among other limitations, ``keylen`` cannot be larger
        than the digest size of the specified hash.
    """
    const, digest_size, block_size = lookup_hash(digest)
    secret = to_bytes(secret, param='secret')
    salt = to_bytes(salt, param='salt')
    if not isinstance(rounds, int_types):
        raise exc.ExpectedTypeError(rounds, 'int', 'rounds')
    if rounds < 1:
        raise ValueError('rounds must be at least 1')
    if keylen is None:
        keylen = digest_size
    elif not isinstance(keylen, int_types):
        raise exc.ExpectedTypeError(keylen, 'int or None', 'keylen')
    elif keylen < 0:
        raise ValueError('keylen must be at least 0')
    elif keylen > digest_size:
        raise ValueError('keylength too large for digest: %r > %r' % (keylen, digest_size))
    block = secret + salt
    for _ in irange(rounds):
        block = const(block).digest()
    return block[:keylen]