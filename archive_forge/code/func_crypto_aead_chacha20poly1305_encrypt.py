from typing import Optional
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_aead_chacha20poly1305_encrypt(message: bytes, aad: Optional[bytes], nonce: bytes, key: bytes) -> bytes:
    """
    Encrypt the given ``message`` using the "legacy" construction
    described in draft-agl-tls-chacha20poly1305.

    :param message:
    :type message: bytes
    :param aad:
    :type aad: Optional[bytes]
    :param nonce:
    :type nonce: bytes
    :param key:
    :type key: bytes
    :return: authenticated ciphertext
    :rtype: bytes
    """
    ensure(isinstance(message, bytes), 'Input message type must be bytes', raising=exc.TypeError)
    mlen = len(message)
    ensure(mlen <= crypto_aead_chacha20poly1305_MESSAGEBYTES_MAX, 'Message must be at most {} bytes long'.format(crypto_aead_chacha20poly1305_MESSAGEBYTES_MAX), raising=exc.ValueError)
    ensure(isinstance(aad, bytes) or aad is None, 'Additional data must be bytes or None', raising=exc.TypeError)
    ensure(isinstance(nonce, bytes) and len(nonce) == crypto_aead_chacha20poly1305_NPUBBYTES, 'Nonce must be a {} bytes long bytes sequence'.format(crypto_aead_chacha20poly1305_NPUBBYTES), raising=exc.TypeError)
    ensure(isinstance(key, bytes) and len(key) == crypto_aead_chacha20poly1305_KEYBYTES, 'Key must be a {} bytes long bytes sequence'.format(crypto_aead_chacha20poly1305_KEYBYTES), raising=exc.TypeError)
    if aad:
        _aad = aad
        aalen = len(aad)
    else:
        _aad = ffi.NULL
        aalen = 0
    mlen = len(message)
    mxout = mlen + crypto_aead_chacha20poly1305_ietf_ABYTES
    clen = ffi.new('unsigned long long *')
    ciphertext = ffi.new('unsigned char[]', mxout)
    res = lib.crypto_aead_chacha20poly1305_encrypt(ciphertext, clen, message, mlen, _aad, aalen, ffi.NULL, nonce, key)
    ensure(res == 0, 'Encryption failed.', raising=exc.CryptoError)
    return ffi.buffer(ciphertext, clen[0])[:]