import sys
from typing import Tuple
import nacl.exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_pwhash_scryptsalsa208sha256_ll(passwd: bytes, salt: bytes, n: int, r: int, p: int, dklen: int=64, maxmem: int=SCRYPT_MAX_MEM) -> bytes:
    """
    Derive a cryptographic key using the ``passwd`` and ``salt``
    given as input.

    The work factor can be tuned by by picking different
    values for the parameters

    :param bytes passwd:
    :param bytes salt:
    :param bytes salt: *must* be *exactly* :py:const:`.SALTBYTES` long
    :param int dklen:
    :param int opslimit:
    :param int n:
    :param int r: block size,
    :param int p: the parallelism factor
    :param int maxmem: the maximum available memory available for scrypt's
                       operations
    :rtype: bytes
    :raises nacl.exceptions.UnavailableError: If called when using a
        minimal build of libsodium.
    """
    ensure(has_crypto_pwhash_scryptsalsa208sha256, 'Not available in minimal build', raising=exc.UnavailableError)
    ensure(isinstance(n, int), raising=TypeError)
    ensure(isinstance(r, int), raising=TypeError)
    ensure(isinstance(p, int), raising=TypeError)
    ensure(isinstance(passwd, bytes), raising=TypeError)
    ensure(isinstance(salt, bytes), raising=TypeError)
    _check_memory_occupation(n, r, p, maxmem)
    buf = ffi.new('uint8_t[]', dklen)
    ret = lib.crypto_pwhash_scryptsalsa208sha256_ll(passwd, len(passwd), salt, len(salt), n, r, p, buf, dklen)
    ensure(ret == 0, 'Unexpected failure in key derivation', raising=exc.RuntimeError)
    return ffi.buffer(ffi.cast('char *', buf), dklen)[:]