from __future__ import annotations
import threading
import typing
from cryptography.exceptions import (
from cryptography.hazmat.backends.openssl.utils import (
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
from cryptography.hazmat.primitives.asymmetric.padding import (
from cryptography.hazmat.primitives.asymmetric.rsa import (
def recover_data_from_signature(self, signature: bytes, padding: AsymmetricPadding, algorithm: typing.Optional[hashes.HashAlgorithm]) -> bytes:
    if isinstance(algorithm, asym_utils.Prehashed):
        raise TypeError('Prehashed is only supported in the sign and verify methods. It cannot be used with recover_data_from_signature.')
    return _rsa_sig_recover(self._backend, padding, algorithm, self, signature)