from typing import Tuple
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_kx_seed_keypair(seed: bytes) -> Tuple[bytes, bytes]:
    """
    Generate a keypair with a given seed.
    This is functionally the same as crypto_box_seed_keypair, however
    it uses the blake2b hash primitive instead of sha512.
    It is included mainly for api consistency when using crypto_kx.
    :param seed: random seed
    :type seed: bytes
    :return: (public_key, secret_key)
    :rtype: (bytes, bytes)
    """
    public_key = ffi.new('unsigned char[]', crypto_kx_PUBLIC_KEY_BYTES)
    secret_key = ffi.new('unsigned char[]', crypto_kx_SECRET_KEY_BYTES)
    ensure(isinstance(seed, bytes) and len(seed) == crypto_kx_SEED_BYTES, 'Seed must be a {} byte long bytes sequence'.format(crypto_kx_SEED_BYTES), raising=exc.TypeError)
    res = lib.crypto_kx_seed_keypair(public_key, secret_key, seed)
    ensure(res == 0, 'Key generation failed.', raising=exc.CryptoError)
    return (ffi.buffer(public_key, crypto_kx_PUBLIC_KEY_BYTES)[:], ffi.buffer(secret_key, crypto_kx_SECRET_KEY_BYTES)[:])