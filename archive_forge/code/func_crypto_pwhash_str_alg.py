import sys
from typing import Tuple
import nacl.exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_pwhash_str_alg(passwd: bytes, opslimit: int, memlimit: int, alg: int) -> bytes:
    """
    Derive a cryptographic key using the ``passwd`` given as input
    and a random salt, returning a string representation which
    includes the salt, the tuning parameters and the used algorithm.

    :param passwd: The input password
    :type passwd: bytes
    :param opslimit: computational cost
    :type opslimit: int
    :param memlimit: memory cost
    :type memlimit: int
    :param alg: The algorithm to use
    :type alg: int
    :return: serialized derived key and parameters
    :rtype: bytes
    """
    ensure(isinstance(opslimit, int), raising=TypeError)
    ensure(isinstance(memlimit, int), raising=TypeError)
    ensure(isinstance(passwd, bytes), raising=TypeError)
    _check_argon2_limits_alg(opslimit, memlimit, alg)
    outbuf = ffi.new('char[]', 128)
    ret = lib.crypto_pwhash_str_alg(outbuf, passwd, len(passwd), opslimit, memlimit, alg)
    ensure(ret == 0, 'Unexpected failure in key derivation', raising=exc.RuntimeError)
    return ffi.string(outbuf)