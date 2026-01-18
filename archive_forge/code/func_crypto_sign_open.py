from typing import Tuple
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_sign_open(signed: bytes, pk: bytes) -> bytes:
    """
    Verifies the signature of the signed message ``signed`` using the public
    key ``pk`` and returns the unsigned message.

    :param signed: bytes
    :param pk: bytes
    :rtype: bytes
    """
    message = ffi.new('unsigned char[]', len(signed))
    message_len = ffi.new('unsigned long long *')
    if lib.crypto_sign_open(message, message_len, signed, len(signed), pk) != 0:
        raise exc.BadSignatureError('Signature was forged or corrupt')
    return ffi.buffer(message, message_len[0])[:]