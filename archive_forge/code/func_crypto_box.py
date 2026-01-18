from typing import Tuple
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_box(message: bytes, nonce: bytes, pk: bytes, sk: bytes) -> bytes:
    """
    Encrypts and returns a message ``message`` using the secret key ``sk``,
    public key ``pk``, and the nonce ``nonce``.

    :param message: bytes
    :param nonce: bytes
    :param pk: bytes
    :param sk: bytes
    :rtype: bytes
    """
    if len(nonce) != crypto_box_NONCEBYTES:
        raise exc.ValueError('Invalid nonce size')
    if len(pk) != crypto_box_PUBLICKEYBYTES:
        raise exc.ValueError('Invalid public key')
    if len(sk) != crypto_box_SECRETKEYBYTES:
        raise exc.ValueError('Invalid secret key')
    padded = b'\x00' * crypto_box_ZEROBYTES + message
    ciphertext = ffi.new('unsigned char[]', len(padded))
    rc = lib.crypto_box(ciphertext, padded, len(padded), nonce, pk, sk)
    ensure(rc == 0, 'Unexpected library error', raising=exc.RuntimeError)
    return ffi.buffer(ciphertext, len(padded))[crypto_box_BOXZEROBYTES:]