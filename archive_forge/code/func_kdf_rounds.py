from __future__ import annotations
import abc
import typing
from cryptography import utils
from cryptography.hazmat.primitives.hashes import HashAlgorithm
def kdf_rounds(self, rounds: int) -> KeySerializationEncryptionBuilder:
    if self._kdf_rounds is not None:
        raise ValueError('kdf_rounds already set')
    if not isinstance(rounds, int):
        raise TypeError('kdf_rounds must be an integer')
    if rounds < 1:
        raise ValueError('kdf_rounds must be a positive integer')
    return KeySerializationEncryptionBuilder(self._format, _kdf_rounds=rounds, _hmac_hash=self._hmac_hash, _key_cert_algorithm=self._key_cert_algorithm)