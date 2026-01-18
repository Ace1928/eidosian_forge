import sys
from typing import Tuple
import nacl.exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_pwhash_scryptsalsa208sha256_str(passwd: bytes, opslimit: int=SCRYPT_OPSLIMIT_INTERACTIVE, memlimit: int=SCRYPT_MEMLIMIT_INTERACTIVE) -> bytes:
    """
    Derive a cryptographic key using the ``passwd`` and ``salt``
    given as input, returning a string representation which includes
    the salt and the tuning parameters.

    The returned string can be directly stored as a password hash.

    See :py:func:`.crypto_pwhash_scryptsalsa208sha256` for a short
    discussion about ``opslimit`` and ``memlimit`` values.

    :param bytes passwd:
    :param int opslimit:
    :param int memlimit:
    :return: serialized key hash, including salt and tuning parameters
    :rtype: bytes
    :raises nacl.exceptions.UnavailableError: If called when using a
        minimal build of libsodium.
    """
    ensure(has_crypto_pwhash_scryptsalsa208sha256, 'Not available in minimal build', raising=exc.UnavailableError)
    buf = ffi.new('char[]', SCRYPT_STRBYTES)
    ret = lib.crypto_pwhash_scryptsalsa208sha256_str(buf, passwd, len(passwd), opslimit, memlimit)
    ensure(ret == 0, 'Unexpected failure in password hashing', raising=exc.RuntimeError)
    return ffi.string(buf)