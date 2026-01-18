import sys
from typing import Tuple
import nacl.exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_pwhash_str_verify(passwd_hash: bytes, passwd: bytes) -> bool:
    """
    Verifies the ``passwd`` against a given password hash.

    Returns True on success, raises InvalidkeyError on failure
    :param passwd_hash: saved password hash
    :type passwd_hash: bytes
    :param passwd: password to be checked
    :type passwd: bytes
    :return: success
    :rtype: boolean
    """
    ensure(isinstance(passwd_hash, bytes), raising=TypeError)
    ensure(isinstance(passwd, bytes), raising=TypeError)
    ensure(len(passwd_hash) <= 127, 'Hash must be at most 127 bytes long', raising=exc.ValueError)
    ret = lib.crypto_pwhash_str_verify(passwd_hash, passwd, len(passwd))
    ensure(ret == 0, 'Wrong password', raising=exc.InvalidkeyError)
    return True