import nacl.bindings
import nacl.encoding
from . import _argon2
def kdf(size: int, password: bytes, salt: bytes, opslimit: int=OPSLIMIT_SENSITIVE, memlimit: int=MEMLIMIT_SENSITIVE, encoder: nacl.encoding.Encoder=nacl.encoding.RawEncoder) -> bytes:
    """
    Derive a ``size`` bytes long key from a caller-supplied
    ``password`` and ``salt`` pair using the argon2i
    memory-hard construct.

    the enclosing module provides the constants

        - :py:const:`.OPSLIMIT_INTERACTIVE`
        - :py:const:`.MEMLIMIT_INTERACTIVE`
        - :py:const:`.OPSLIMIT_MODERATE`
        - :py:const:`.MEMLIMIT_MODERATE`
        - :py:const:`.OPSLIMIT_SENSITIVE`
        - :py:const:`.MEMLIMIT_SENSITIVE`

    as a guidance for correct settings.

    :param size: derived key size, must be between
                 :py:const:`.BYTES_MIN` and
                 :py:const:`.BYTES_MAX`
    :type size: int
    :param password: password used to seed the key derivation procedure;
                     it length must be between
                     :py:const:`.PASSWD_MIN` and
                     :py:const:`.PASSWD_MAX`
    :type password: bytes
    :param salt: **RANDOM** salt used in the key derivation procedure;
                 its length must be exactly :py:const:`.SALTBYTES`
    :type salt: bytes
    :param opslimit: the time component (operation count)
                     of the key derivation procedure's computational cost;
                     it must be between
                     :py:const:`.OPSLIMIT_MIN` and
                     :py:const:`.OPSLIMIT_MAX`
    :type opslimit: int
    :param memlimit: the memory occupation component
                     of the key derivation procedure's computational cost;
                     it must be between
                     :py:const:`.MEMLIMIT_MIN` and
                     :py:const:`.MEMLIMIT_MAX`
    :type memlimit: int
    :rtype: bytes

    .. versionadded:: 1.2
    """
    return encoder.encode(nacl.bindings.crypto_pwhash_alg(size, password, salt, opslimit, memlimit, ALG))