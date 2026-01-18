from typing import Tuple
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_box_seal(message: bytes, pk: bytes) -> bytes:
    """
    Encrypts and returns a message ``message`` using an ephemeral secret key
    and the public key ``pk``.
    The ephemeral public key, which is embedded in the sealed box, is also
    used, in combination with ``pk``, to derive the nonce needed for the
    underlying box construct.

    :param message: bytes
    :param pk: bytes
    :rtype: bytes

    .. versionadded:: 1.2
    """
    ensure(isinstance(message, bytes), 'input message must be bytes', raising=TypeError)
    ensure(isinstance(pk, bytes), 'public key must be bytes', raising=TypeError)
    if len(pk) != crypto_box_PUBLICKEYBYTES:
        raise exc.ValueError('Invalid public key')
    _mlen = len(message)
    _clen = crypto_box_SEALBYTES + _mlen
    ciphertext = ffi.new('unsigned char[]', _clen)
    rc = lib.crypto_box_seal(ciphertext, message, _mlen, pk)
    ensure(rc == 0, 'Unexpected library error', raising=exc.RuntimeError)
    return ffi.buffer(ciphertext, _clen)[:]