from typing import ByteString, Optional, Tuple, cast
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_secretstream_xchacha20poly1305_init_pull(state: crypto_secretstream_xchacha20poly1305_state, header: bytes, key: bytes) -> None:
    """
    Initialize a crypto_secretstream_xchacha20poly1305 decryption buffer.

    :param state: a secretstream state object
    :type state: crypto_secretstream_xchacha20poly1305_state
    :param header: must be
                :data:`.crypto_secretstream_xchacha20poly1305_HEADERBYTES` long
    :type header: bytes
    :param key: must be
                :data:`.crypto_secretstream_xchacha20poly1305_KEYBYTES` long
    :type key: bytes

    """
    ensure(isinstance(state, crypto_secretstream_xchacha20poly1305_state), 'State must be a crypto_secretstream_xchacha20poly1305_state object', raising=exc.TypeError)
    ensure(isinstance(header, bytes), 'Header must be a bytes sequence', raising=exc.TypeError)
    ensure(len(header) == crypto_secretstream_xchacha20poly1305_HEADERBYTES, 'Invalid header length', raising=exc.ValueError)
    ensure(isinstance(key, bytes), 'Key must be a bytes sequence', raising=exc.TypeError)
    ensure(len(key) == crypto_secretstream_xchacha20poly1305_KEYBYTES, 'Invalid key length', raising=exc.ValueError)
    if state.tagbuf is None:
        state.tagbuf = ffi.new('unsigned char *')
    rc = lib.crypto_secretstream_xchacha20poly1305_init_pull(state.statebuf, header, key)
    ensure(rc == 0, 'Unexpected failure', raising=exc.RuntimeError)