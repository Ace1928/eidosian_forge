from __future__ import annotations
import abc
from cryptography.exceptions import UnsupportedAlgorithm, _Reasons
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
from cryptography.hazmat.primitives import _serialization
@abc.abstractmethod
def private_bytes_raw(self) -> bytes:
    """
        The raw bytes of the private key.
        Equivalent to private_bytes(Raw, Raw, NoEncryption()).
        """