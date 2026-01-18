from typing import Tuple
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_sign(message: bytes, sk: bytes) -> bytes:
    """
    Signs the message ``message`` using the secret key ``sk`` and returns the
    signed message.

    :param message: bytes
    :param sk: bytes
    :rtype: bytes
    """
    signed = ffi.new('unsigned char[]', len(message) + crypto_sign_BYTES)
    signed_len = ffi.new('unsigned long long *')
    rc = lib.crypto_sign(signed, signed_len, message, len(message), sk)
    ensure(rc == 0, 'Unexpected library error', raising=exc.RuntimeError)
    return ffi.buffer(signed, signed_len[0])[:]