from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def sodium_init() -> None:
    """
    Initializes sodium, picking the best implementations available for this
    machine.
    """
    ffi.init_once(_sodium_init, 'libsodium')