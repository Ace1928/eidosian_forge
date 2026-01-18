import sys
from typing import Tuple
import nacl.exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_pwhash_scryptsalsa208sha256_str_verify(passwd_hash: bytes, passwd: bytes) -> bool:
    """
    Verifies the ``passwd`` against the ``passwd_hash`` that was generated.
    Returns True or False depending on the success

    :param passwd_hash: bytes
    :param passwd: bytes
    :rtype: boolean
    :raises nacl.exceptions.UnavailableError: If called when using a
        minimal build of libsodium.
    """
    ensure(has_crypto_pwhash_scryptsalsa208sha256, 'Not available in minimal build', raising=exc.UnavailableError)
    ensure(len(passwd_hash) == SCRYPT_STRBYTES - 1, 'Invalid password hash', raising=exc.ValueError)
    ret = lib.crypto_pwhash_scryptsalsa208sha256_str_verify(passwd_hash, passwd, len(passwd))
    ensure(ret == 0, 'Wrong password', raising=exc.InvalidkeyError)
    return True