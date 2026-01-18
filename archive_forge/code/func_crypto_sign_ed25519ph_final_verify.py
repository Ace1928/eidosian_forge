from typing import Tuple
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_sign_ed25519ph_final_verify(edph: crypto_sign_ed25519ph_state, signature: bytes, pk: bytes) -> bool:
    """
    Verify a prehashed signature using the public key pk

    :param edph: the ed25519ph state for the data
                 being verified
    :type edph: crypto_sign_ed25519ph_state
    :param signature: the signature being verified
    :type signature: bytes
    :param pk: the ed25519 public part of the signing key
    :type pk: bytes
    :return: True if the signature is valid
    :rtype: boolean
    :raises exc.BadSignatureError: if the signature is not valid
    """
    ensure(isinstance(edph, crypto_sign_ed25519ph_state), 'edph parameter must be a ed25519ph_state object', raising=exc.TypeError)
    ensure(isinstance(signature, bytes), 'signature parameter must be a bytes object', raising=exc.TypeError)
    ensure(len(signature) == crypto_sign_BYTES, 'signature must be {} bytes long'.format(crypto_sign_BYTES), raising=exc.TypeError)
    ensure(isinstance(pk, bytes), 'public key parameter must be a bytes object', raising=exc.TypeError)
    ensure(len(pk) == crypto_sign_PUBLICKEYBYTES, 'public key must be {} bytes long'.format(crypto_sign_PUBLICKEYBYTES), raising=exc.TypeError)
    rc = lib.crypto_sign_ed25519ph_final_verify(edph.state, signature, pk)
    if rc != 0:
        raise exc.BadSignatureError('Signature was forged or corrupt')
    return True