from __future__ import annotations
import typing
from cryptography.hazmat.primitives.ciphers import Cipher
from cryptography.hazmat.primitives.ciphers.algorithms import AES
from cryptography.hazmat.primitives.ciphers.modes import ECB
from cryptography.hazmat.primitives.constant_time import bytes_eq
def aes_key_unwrap(wrapping_key: bytes, wrapped_key: bytes, backend: typing.Any=None) -> bytes:
    if len(wrapped_key) < 24:
        raise InvalidUnwrap('Must be at least 24 bytes')
    if len(wrapped_key) % 8 != 0:
        raise InvalidUnwrap('The wrapped key must be a multiple of 8 bytes')
    if len(wrapping_key) not in [16, 24, 32]:
        raise ValueError('The wrapping key must be a valid AES key length')
    aiv = b'\xa6\xa6\xa6\xa6\xa6\xa6\xa6\xa6'
    r = [wrapped_key[i:i + 8] for i in range(0, len(wrapped_key), 8)]
    a = r.pop(0)
    a, r = _unwrap_core(wrapping_key, a, r)
    if not bytes_eq(a, aiv):
        raise InvalidUnwrap()
    return b''.join(r)