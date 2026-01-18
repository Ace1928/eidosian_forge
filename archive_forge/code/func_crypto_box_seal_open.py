from typing import Tuple
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_box_seal_open(ciphertext: bytes, pk: bytes, sk: bytes) -> bytes:
    """
    Decrypts and returns an encrypted message ``ciphertext``, using the
    recipent's secret key ``sk`` and the sender's ephemeral public key
    embedded in the sealed box. The box contruct nonce is derived from
    the recipient's public key ``pk`` and the sender's public key.

    :param ciphertext: bytes
    :param pk: bytes
    :param sk: bytes
    :rtype: bytes

    .. versionadded:: 1.2
    """
    ensure(isinstance(ciphertext, bytes), 'input ciphertext must be bytes', raising=TypeError)
    ensure(isinstance(pk, bytes), 'public key must be bytes', raising=TypeError)
    ensure(isinstance(sk, bytes), 'secret key must be bytes', raising=TypeError)
    if len(pk) != crypto_box_PUBLICKEYBYTES:
        raise exc.ValueError('Invalid public key')
    if len(sk) != crypto_box_SECRETKEYBYTES:
        raise exc.ValueError('Invalid secret key')
    _clen = len(ciphertext)
    ensure(_clen >= crypto_box_SEALBYTES, 'Input cyphertext must be at least {} long'.format(crypto_box_SEALBYTES), raising=exc.TypeError)
    _mlen = _clen - crypto_box_SEALBYTES
    plaintext = ffi.new('unsigned char[]', max(1, _mlen))
    res = lib.crypto_box_seal_open(plaintext, ciphertext, _clen, pk, sk)
    ensure(res == 0, 'An error occurred trying to decrypt the message', raising=exc.CryptoError)
    return ffi.buffer(plaintext, _mlen)[:]