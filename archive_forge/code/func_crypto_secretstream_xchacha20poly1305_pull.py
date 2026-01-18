from typing import ByteString, Optional, Tuple, cast
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_secretstream_xchacha20poly1305_pull(state: crypto_secretstream_xchacha20poly1305_state, c: bytes, ad: Optional[bytes]=None) -> Tuple[bytes, int]:
    """
    Read a decrypted message from the secret stream.

    :param state: a secretstream state object
    :type state: crypto_secretstream_xchacha20poly1305_state
    :param c: the ciphertext to decrypt, the maximum length of an individual
              ciphertext is
              :data:`.crypto_secretstream_xchacha20poly1305_MESSAGEBYTES_MAX` +
              :data:`.crypto_secretstream_xchacha20poly1305_ABYTES`.
    :type c: bytes
    :param ad: additional data to include in the authentication tag
    :type ad: bytes or None
    :return: (message, tag)
    :rtype: (bytes, int)

    """
    ensure(isinstance(state, crypto_secretstream_xchacha20poly1305_state), 'State must be a crypto_secretstream_xchacha20poly1305_state object', raising=exc.TypeError)
    ensure(state.tagbuf is not None, 'State must be initialized using crypto_secretstream_xchacha20poly1305_init_pull', raising=exc.ValueError)
    ensure(isinstance(c, bytes), 'Ciphertext is not bytes', raising=exc.TypeError)
    ensure(len(c) >= crypto_secretstream_xchacha20poly1305_ABYTES, 'Ciphertext is too short', raising=exc.ValueError)
    ensure(len(c) <= crypto_secretstream_xchacha20poly1305_MESSAGEBYTES_MAX + crypto_secretstream_xchacha20poly1305_ABYTES, 'Ciphertext is too long', raising=exc.ValueError)
    ensure(ad is None or isinstance(ad, bytes), 'Additional data must be bytes or None', raising=exc.TypeError)
    mlen = len(c) - crypto_secretstream_xchacha20poly1305_ABYTES
    if state.rawbuf is None or len(state.rawbuf) < mlen:
        state.rawbuf = ffi.new('unsigned char[]', mlen)
    if ad is None:
        ad = ffi.NULL
        adlen = 0
    else:
        adlen = len(ad)
    rc = lib.crypto_secretstream_xchacha20poly1305_pull(state.statebuf, state.rawbuf, ffi.NULL, state.tagbuf, c, len(c), ad, adlen)
    ensure(rc == 0, 'Unexpected failure', raising=exc.RuntimeError)
    return (ffi.buffer(state.rawbuf, mlen)[:], int(cast(bytes, state.tagbuf)[0]))