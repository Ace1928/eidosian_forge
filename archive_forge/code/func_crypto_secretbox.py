from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_secretbox(message: bytes, nonce: bytes, key: bytes) -> bytes:
    """
    Encrypts and returns the message ``message`` with the secret ``key`` and
    the nonce ``nonce``.

    :param message: bytes
    :param nonce: bytes
    :param key: bytes
    :rtype: bytes
    """
    if len(key) != crypto_secretbox_KEYBYTES:
        raise exc.ValueError('Invalid key')
    if len(nonce) != crypto_secretbox_NONCEBYTES:
        raise exc.ValueError('Invalid nonce')
    padded = b'\x00' * crypto_secretbox_ZEROBYTES + message
    ciphertext = ffi.new('unsigned char[]', len(padded))
    res = lib.crypto_secretbox(ciphertext, padded, len(padded), nonce, key)
    ensure(res == 0, 'Encryption failed', raising=exc.CryptoError)
    ciphertext = ffi.buffer(ciphertext, len(padded))
    return ciphertext[crypto_secretbox_BOXZEROBYTES:]