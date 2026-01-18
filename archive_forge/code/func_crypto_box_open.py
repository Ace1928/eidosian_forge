from typing import Tuple
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_box_open(ciphertext: bytes, nonce: bytes, pk: bytes, sk: bytes) -> bytes:
    """
    Decrypts and returns an encrypted message ``ciphertext``, using the secret
    key ``sk``, public key ``pk``, and the nonce ``nonce``.

    :param ciphertext: bytes
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
    padded = b'\x00' * crypto_box_BOXZEROBYTES + ciphertext
    plaintext = ffi.new('unsigned char[]', len(padded))
    res = lib.crypto_box_open(plaintext, padded, len(padded), nonce, pk, sk)
    ensure(res == 0, 'An error occurred trying to decrypt the message', raising=exc.CryptoError)
    return ffi.buffer(plaintext, len(padded))[crypto_box_ZEROBYTES:]