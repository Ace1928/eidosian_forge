from typing import Tuple
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_sign_seed_keypair(seed: bytes) -> Tuple[bytes, bytes]:
    """
    Computes and returns the public key and secret key using the seed ``seed``.

    :param seed: bytes
    :rtype: (bytes(public_key), bytes(secret_key))
    """
    if len(seed) != crypto_sign_SEEDBYTES:
        raise exc.ValueError('Invalid seed')
    pk = ffi.new('unsigned char[]', crypto_sign_PUBLICKEYBYTES)
    sk = ffi.new('unsigned char[]', crypto_sign_SECRETKEYBYTES)
    rc = lib.crypto_sign_seed_keypair(pk, sk, seed)
    ensure(rc == 0, 'Unexpected library error', raising=exc.RuntimeError)
    return (ffi.buffer(pk, crypto_sign_PUBLICKEYBYTES)[:], ffi.buffer(sk, crypto_sign_SECRETKEYBYTES)[:])