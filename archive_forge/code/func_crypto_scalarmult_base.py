from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_scalarmult_base(n: bytes) -> bytes:
    """
    Computes and returns the scalar product of a standard group element and an
    integer ``n``.

    :param n: bytes
    :rtype: bytes
    """
    q = ffi.new('unsigned char[]', crypto_scalarmult_BYTES)
    rc = lib.crypto_scalarmult_base(q, n)
    ensure(rc == 0, 'Unexpected library error', raising=exc.RuntimeError)
    return ffi.buffer(q, crypto_scalarmult_SCALARBYTES)[:]