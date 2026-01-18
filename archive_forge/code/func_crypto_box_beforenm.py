from typing import Tuple
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_box_beforenm(pk: bytes, sk: bytes) -> bytes:
    """
    Computes and returns the shared key for the public key ``pk`` and the
    secret key ``sk``. This can be used to speed up operations where the same
    set of keys is going to be used multiple times.

    :param pk: bytes
    :param sk: bytes
    :rtype: bytes
    """
    if len(pk) != crypto_box_PUBLICKEYBYTES:
        raise exc.ValueError('Invalid public key')
    if len(sk) != crypto_box_SECRETKEYBYTES:
        raise exc.ValueError('Invalid secret key')
    k = ffi.new('unsigned char[]', crypto_box_BEFORENMBYTES)
    rc = lib.crypto_box_beforenm(k, pk, sk)
    ensure(rc == 0, 'Unexpected library error', raising=exc.RuntimeError)
    return ffi.buffer(k, crypto_box_BEFORENMBYTES)[:]