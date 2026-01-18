from __future__ import annotations
import abc
import typing
from cryptography import utils
from cryptography.hazmat.primitives.hashes import HashAlgorithm
def key_cert_algorithm(self, algorithm: PBES) -> KeySerializationEncryptionBuilder:
    if self._format is not PrivateFormat.PKCS12:
        raise TypeError('key_cert_algorithm only supported with PrivateFormat.PKCS12')
    if self._key_cert_algorithm is not None:
        raise ValueError('key_cert_algorithm already set')
    return KeySerializationEncryptionBuilder(self._format, _kdf_rounds=self._kdf_rounds, _hmac_hash=self._hmac_hash, _key_cert_algorithm=algorithm)