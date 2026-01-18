from __future__ import annotations
import abc
import typing
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
@property
@abc.abstractmethod
def block_size(self) -> typing.Optional[int]:
    """
        The internal block size of the hash function, or None if the hash
        function does not use blocks internally (e.g. SHA3).
        """