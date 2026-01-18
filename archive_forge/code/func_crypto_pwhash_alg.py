import sys
from typing import Tuple
import nacl.exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_pwhash_alg(outlen: int, passwd: bytes, salt: bytes, opslimit: int, memlimit: int, alg: int) -> bytes:
    """
    Derive a raw cryptographic key using the ``passwd`` and the ``salt``
    given as input to the ``alg`` algorithm.

    :param outlen: the length of the derived key
    :type outlen: int
    :param passwd: The input password
    :type passwd: bytes
    :param salt:
    :type salt: bytes
    :param opslimit: computational cost
    :type opslimit: int
    :param memlimit: memory cost
    :type memlimit: int
    :param alg: algorithm identifier
    :type alg: int
    :return: derived key
    :rtype: bytes
    """
    ensure(isinstance(outlen, int), raising=exc.TypeError)
    ensure(isinstance(opslimit, int), raising=exc.TypeError)
    ensure(isinstance(memlimit, int), raising=exc.TypeError)
    ensure(isinstance(alg, int), raising=exc.TypeError)
    ensure(isinstance(passwd, bytes), raising=exc.TypeError)
    if len(salt) != crypto_pwhash_SALTBYTES:
        raise exc.ValueError('salt must be exactly {} bytes long'.format(crypto_pwhash_SALTBYTES))
    if outlen < crypto_pwhash_BYTES_MIN:
        raise exc.ValueError('derived key must be at least {} bytes long'.format(crypto_pwhash_BYTES_MIN))
    elif outlen > crypto_pwhash_BYTES_MAX:
        raise exc.ValueError('derived key must be at most {} bytes long'.format(crypto_pwhash_BYTES_MAX))
    _check_argon2_limits_alg(opslimit, memlimit, alg)
    outbuf = ffi.new('unsigned char[]', outlen)
    ret = lib.crypto_pwhash(outbuf, outlen, passwd, len(passwd), salt, opslimit, memlimit, alg)
    ensure(ret == 0, 'Unexpected failure in key derivation', raising=exc.RuntimeError)
    return ffi.buffer(outbuf, outlen)[:]