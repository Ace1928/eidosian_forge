from typing import Tuple
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_box_afternm(message: bytes, nonce: bytes, k: bytes) -> bytes:
    """
    Encrypts and returns the message ``message`` using the shared key ``k`` and
    the nonce ``nonce``.

    :param message: bytes
    :param nonce: bytes
    :param k: bytes
    :rtype: bytes
    """
    if len(nonce) != crypto_box_NONCEBYTES:
        raise exc.ValueError('Invalid nonce')
    if len(k) != crypto_box_BEFORENMBYTES:
        raise exc.ValueError('Invalid shared key')
    padded = b'\x00' * crypto_box_ZEROBYTES + message
    ciphertext = ffi.new('unsigned char[]', len(padded))
    rc = lib.crypto_box_afternm(ciphertext, padded, len(padded), nonce, k)
    ensure(rc == 0, 'Unexpected library error', raising=exc.RuntimeError)
    return ffi.buffer(ciphertext, len(padded))[crypto_box_BOXZEROBYTES:]