from typing import NoReturn, TypeVar
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def generichash_blake2b_update(state: Blake2State, data: bytes) -> None:
    """Update the blake2b hash state

    :param state: a initialized Blake2bState object as returned from
                     :py:func:`.crypto_generichash_blake2b_init`
    :type state: :py:class:`.Blake2State`
    :param data:
    :type data: bytes
    """
    ensure(isinstance(state, Blake2State), 'State must be a Blake2State object', raising=exc.TypeError)
    ensure(isinstance(data, bytes), 'Input data must be a bytes sequence', raising=exc.TypeError)
    rc = lib.crypto_generichash_blake2b_update(state._statebuf, data, len(data))
    ensure(rc == 0, 'Unexpected failure', raising=exc.RuntimeError)