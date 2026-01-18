from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_scalarmult(n: bytes, p: bytes) -> bytes:
    """
    Computes and returns the scalar product of the given group element and an
    integer ``n``.

    :param p: bytes
    :param n: bytes
    :rtype: bytes
    """
    q = ffi.new('unsigned char[]', crypto_scalarmult_BYTES)
    rc = lib.crypto_scalarmult(q, n, p)
    ensure(rc == 0, 'Unexpected library error', raising=exc.RuntimeError)
    return ffi.buffer(q, crypto_scalarmult_SCALARBYTES)[:]