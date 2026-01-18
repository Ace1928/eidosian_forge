from typing import Tuple
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_box_keypair() -> Tuple[bytes, bytes]:
    """
    Returns a randomly generated public and secret key.

    :rtype: (bytes(public_key), bytes(secret_key))
    """
    pk = ffi.new('unsigned char[]', crypto_box_PUBLICKEYBYTES)
    sk = ffi.new('unsigned char[]', crypto_box_SECRETKEYBYTES)
    rc = lib.crypto_box_keypair(pk, sk)
    ensure(rc == 0, 'Unexpected library error', raising=exc.RuntimeError)
    return (ffi.buffer(pk, crypto_box_PUBLICKEYBYTES)[:], ffi.buffer(sk, crypto_box_SECRETKEYBYTES)[:])