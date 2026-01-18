from __future__ import annotations
import abc
import typing
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives._asymmetric import (
from cryptography.hazmat.primitives.asymmetric import rsa
def calculate_max_pss_salt_length(key: typing.Union[rsa.RSAPrivateKey, rsa.RSAPublicKey], hash_algorithm: hashes.HashAlgorithm) -> int:
    if not isinstance(key, (rsa.RSAPrivateKey, rsa.RSAPublicKey)):
        raise TypeError('key must be an RSA public or private key')
    emlen = (key.key_size + 6) // 8
    salt_length = emlen - hash_algorithm.digest_size - 2
    assert salt_length >= 0
    return salt_length