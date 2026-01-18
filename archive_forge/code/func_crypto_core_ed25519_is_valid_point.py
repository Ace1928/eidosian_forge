from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_core_ed25519_is_valid_point(p: bytes) -> bool:
    """
    Check if ``p`` represents a point on the edwards25519 curve, in canonical
    form, on the main subgroup, and that the point doesn't have a small order.

    :param p: a :py:data:`.crypto_core_ed25519_BYTES` long bytes sequence
              representing a point on the edwards25519 curve
    :type p: bytes
    :return: point validity
    :rtype: bool
    :raises nacl.exceptions.UnavailableError: If called when using a
        minimal build of libsodium.
    """
    ensure(has_crypto_core_ed25519, 'Not available in minimal build', raising=exc.UnavailableError)
    ensure(isinstance(p, bytes) and len(p) == crypto_core_ed25519_BYTES, 'Point must be a crypto_core_ed25519_BYTES long bytes sequence', raising=exc.TypeError)
    rc = lib.crypto_core_ed25519_is_valid_point(p)
    return rc == 1