from typing import Tuple
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_sign_ed25519_sk_to_pk(secret_key_bytes: bytes) -> bytes:
    """
    Extract the public Ed25519 key from a secret Ed25519 key (encoded
    as bytes ``secret_key_bytes``).

    Raises a ValueError if ``secret_key_bytes``is not of length
    ``crypto_sign_SECRETKEYBYTES``

    :param secret_key_bytes: bytes
    :rtype: bytes
    """
    if len(secret_key_bytes) != crypto_sign_SECRETKEYBYTES:
        raise exc.ValueError('Invalid secret key')
    return secret_key_bytes[crypto_sign_SEEDBYTES:]