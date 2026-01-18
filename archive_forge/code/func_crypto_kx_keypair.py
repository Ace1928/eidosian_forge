from typing import Tuple
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_kx_keypair() -> Tuple[bytes, bytes]:
    """
    Generate a keypair.
    This is a duplicate crypto_box_keypair, but
    is included for api consistency.
    :return: (public_key, secret_key)
    :rtype: (bytes, bytes)
    """
    public_key = ffi.new('unsigned char[]', crypto_kx_PUBLIC_KEY_BYTES)
    secret_key = ffi.new('unsigned char[]', crypto_kx_SECRET_KEY_BYTES)
    res = lib.crypto_kx_keypair(public_key, secret_key)
    ensure(res == 0, 'Key generation failed.', raising=exc.CryptoError)
    return (ffi.buffer(public_key, crypto_kx_PUBLIC_KEY_BYTES)[:], ffi.buffer(secret_key, crypto_kx_SECRET_KEY_BYTES)[:])