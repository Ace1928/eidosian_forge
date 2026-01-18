import nacl.exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_shorthash_siphash24(data: bytes, key: bytes) -> bytes:
    """Compute a fast, cryptographic quality, keyed hash of the input data

    :param data:
    :type data: bytes
    :param key: len(key) must be equal to
                :py:data:`.KEYBYTES` (16)
    :type key: bytes
    """
    if len(key) != KEYBYTES:
        raise exc.ValueError('Key length must be exactly {} bytes'.format(KEYBYTES))
    digest = ffi.new('unsigned char[]', BYTES)
    rc = lib.crypto_shorthash_siphash24(digest, data, len(data), key)
    ensure(rc == 0, raising=exc.RuntimeError)
    return ffi.buffer(digest, BYTES)[:]