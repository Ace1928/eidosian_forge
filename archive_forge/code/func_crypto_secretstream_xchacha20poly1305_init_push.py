from typing import ByteString, Optional, Tuple, cast
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_secretstream_xchacha20poly1305_init_push(state: crypto_secretstream_xchacha20poly1305_state, key: bytes) -> bytes:
    """
    Initialize a crypto_secretstream_xchacha20poly1305 encryption buffer.

    :param state: a secretstream state object
    :type state: crypto_secretstream_xchacha20poly1305_state
    :param key: must be
                :data:`.crypto_secretstream_xchacha20poly1305_KEYBYTES` long
    :type key: bytes
    :return: header
    :rtype: bytes

    """
    ensure(isinstance(state, crypto_secretstream_xchacha20poly1305_state), 'State must be a crypto_secretstream_xchacha20poly1305_state object', raising=exc.TypeError)
    ensure(isinstance(key, bytes), 'Key must be a bytes sequence', raising=exc.TypeError)
    ensure(len(key) == crypto_secretstream_xchacha20poly1305_KEYBYTES, 'Invalid key length', raising=exc.ValueError)
    headerbuf = ffi.new('unsigned char []', crypto_secretstream_xchacha20poly1305_HEADERBYTES)
    rc = lib.crypto_secretstream_xchacha20poly1305_init_push(state.statebuf, headerbuf, key)
    ensure(rc == 0, 'Unexpected failure', raising=exc.RuntimeError)
    return ffi.buffer(headerbuf)[:]