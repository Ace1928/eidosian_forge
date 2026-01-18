from typing import Tuple
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_sign_ed25519ph_update(edph: crypto_sign_ed25519ph_state, pmsg: bytes) -> None:
    """
    Update the hash state wrapped in edph

    :param edph: the ed25519ph state being updated
    :type edph: crypto_sign_ed25519ph_state
    :param pmsg: the partial message
    :type pmsg: bytes
    :rtype: None
    """
    ensure(isinstance(edph, crypto_sign_ed25519ph_state), 'edph parameter must be a ed25519ph_state object', raising=exc.TypeError)
    ensure(isinstance(pmsg, bytes), 'pmsg parameter must be a bytes object', raising=exc.TypeError)
    rc = lib.crypto_sign_ed25519ph_update(edph.state, pmsg, len(pmsg))
    ensure(rc == 0, 'Unexpected library error', raising=exc.RuntimeError)