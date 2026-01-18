from __future__ import annotations
import email.base64mime
import email.generator
import email.message
import email.policy
import io
import typing
from cryptography import utils, x509
from cryptography.hazmat.bindings._rust import pkcs7 as rust_pkcs7
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.utils import _check_byteslike
def add_signer(self, certificate: x509.Certificate, private_key: PKCS7PrivateKeyTypes, hash_algorithm: PKCS7HashTypes) -> PKCS7SignatureBuilder:
    if not isinstance(hash_algorithm, (hashes.SHA224, hashes.SHA256, hashes.SHA384, hashes.SHA512)):
        raise TypeError('hash_algorithm must be one of hashes.SHA224, SHA256, SHA384, or SHA512')
    if not isinstance(certificate, x509.Certificate):
        raise TypeError('certificate must be a x509.Certificate')
    if not isinstance(private_key, (rsa.RSAPrivateKey, ec.EllipticCurvePrivateKey)):
        raise TypeError('Only RSA & EC keys are supported at this time.')
    return PKCS7SignatureBuilder(self._data, self._signers + [(certificate, private_key, hash_algorithm)])