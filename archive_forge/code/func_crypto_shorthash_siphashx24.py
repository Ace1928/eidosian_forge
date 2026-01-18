import nacl.exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_shorthash_siphashx24(data: bytes, key: bytes) -> bytes:
    """Compute a fast, cryptographic quality, keyed hash of the input data

    :param data:
    :type data: bytes
    :param key: len(key) must be equal to
                :py:data:`.XKEYBYTES` (16)
    :type key: bytes
    :raises nacl.exceptions.UnavailableError: If called when using a
        minimal build of libsodium.
    """
    ensure(has_crypto_shorthash_siphashx24, 'Not available in minimal build', raising=exc.UnavailableError)
    if len(key) != XKEYBYTES:
        raise exc.ValueError('Key length must be exactly {} bytes'.format(XKEYBYTES))
    digest = ffi.new('unsigned char[]', XBYTES)
    rc = lib.crypto_shorthash_siphashx24(digest, data, len(data), key)
    ensure(rc == 0, raising=exc.RuntimeError)
    return ffi.buffer(digest, XBYTES)[:]