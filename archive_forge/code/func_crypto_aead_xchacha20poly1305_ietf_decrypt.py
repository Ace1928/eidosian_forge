from typing import Optional
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_aead_xchacha20poly1305_ietf_decrypt(ciphertext: bytes, aad: Optional[bytes], nonce: bytes, key: bytes) -> bytes:
    """
    Decrypt the given ``ciphertext`` using the long-nonces xchacha20poly1305
    construction.

    :param ciphertext: authenticated ciphertext
    :type ciphertext: bytes
    :param aad:
    :type aad: Optional[bytes]
    :param nonce:
    :type nonce: bytes
    :param key:
    :type key: bytes
    :return: message
    :rtype: bytes
    """
    ensure(isinstance(ciphertext, bytes), 'Input ciphertext type must be bytes', raising=exc.TypeError)
    clen = len(ciphertext)
    ensure(clen <= _aead_xchacha20poly1305_ietf_CRYPTBYTES_MAX, 'Ciphertext must be at most {} bytes long'.format(_aead_xchacha20poly1305_ietf_CRYPTBYTES_MAX), raising=exc.ValueError)
    ensure(isinstance(aad, bytes) or aad is None, 'Additional data must be bytes or None', raising=exc.TypeError)
    ensure(isinstance(nonce, bytes) and len(nonce) == crypto_aead_xchacha20poly1305_ietf_NPUBBYTES, 'Nonce must be a {} bytes long bytes sequence'.format(crypto_aead_xchacha20poly1305_ietf_NPUBBYTES), raising=exc.TypeError)
    ensure(isinstance(key, bytes) and len(key) == crypto_aead_xchacha20poly1305_ietf_KEYBYTES, 'Key must be a {} bytes long bytes sequence'.format(crypto_aead_xchacha20poly1305_ietf_KEYBYTES), raising=exc.TypeError)
    mxout = clen - crypto_aead_xchacha20poly1305_ietf_ABYTES
    mlen = ffi.new('unsigned long long *')
    message = ffi.new('unsigned char[]', mxout)
    if aad:
        _aad = aad
        aalen = len(aad)
    else:
        _aad = ffi.NULL
        aalen = 0
    res = lib.crypto_aead_xchacha20poly1305_ietf_decrypt(message, mlen, ffi.NULL, ciphertext, clen, _aad, aalen, nonce, key)
    ensure(res == 0, 'Decryption failed.', raising=exc.CryptoError)
    return ffi.buffer(message, mlen[0])[:]