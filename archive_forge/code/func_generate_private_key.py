from __future__ import annotations
import abc
import typing
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
from cryptography.hazmat.primitives import _serialization
@abc.abstractmethod
def generate_private_key(self) -> DHPrivateKey:
    """
        Generates and returns a DHPrivateKey.
        """