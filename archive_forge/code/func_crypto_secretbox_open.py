from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_secretbox_open(ciphertext: bytes, nonce: bytes, key: bytes) -> bytes:
    """
    Decrypt and returns the encrypted message ``ciphertext`` with the secret
    ``key`` and the nonce ``nonce``.

    :param ciphertext: bytes
    :param nonce: bytes
    :param key: bytes
    :rtype: bytes
    """
    if len(key) != crypto_secretbox_KEYBYTES:
        raise exc.ValueError('Invalid key')
    if len(nonce) != crypto_secretbox_NONCEBYTES:
        raise exc.ValueError('Invalid nonce')
    padded = b'\x00' * crypto_secretbox_BOXZEROBYTES + ciphertext
    plaintext = ffi.new('unsigned char[]', len(padded))
    res = lib.crypto_secretbox_open(plaintext, padded, len(padded), nonce, key)
    ensure(res == 0, 'Decryption failed. Ciphertext failed verification', raising=exc.CryptoError)
    plaintext = ffi.buffer(plaintext, len(padded))
    return plaintext[crypto_secretbox_ZEROBYTES:]