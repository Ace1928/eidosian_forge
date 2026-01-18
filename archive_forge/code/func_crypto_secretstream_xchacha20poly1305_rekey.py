from typing import ByteString, Optional, Tuple, cast
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_secretstream_xchacha20poly1305_rekey(state: crypto_secretstream_xchacha20poly1305_state) -> None:
    """
    Explicitly change the encryption key in the stream.

    Normally the stream is re-keyed as needed or an explicit ``tag`` of
    :data:`.crypto_secretstream_xchacha20poly1305_TAG_REKEY` is added to a
    message to ensure forward secrecy, but this method can be used instead
    if the re-keying is controlled without adding the tag.

    :param state: a secretstream state object
    :type state: crypto_secretstream_xchacha20poly1305_state

    """
    ensure(isinstance(state, crypto_secretstream_xchacha20poly1305_state), 'State must be a crypto_secretstream_xchacha20poly1305_state object', raising=exc.TypeError)
    lib.crypto_secretstream_xchacha20poly1305_rekey(state.statebuf)