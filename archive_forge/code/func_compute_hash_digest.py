from __future__ import annotations
import hashlib
import hmac
import json
import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn, Union, cast, overload
from .exceptions import InvalidKeyError
from .types import HashlibHash, JWKDict
from .utils import (
def compute_hash_digest(self, bytestr: bytes) -> bytes:
    """
        Compute a hash digest using the specified algorithm's hash algorithm.

        If there is no hash algorithm, raises a NotImplementedError.
        """
    hash_alg = getattr(self, 'hash_alg', None)
    if hash_alg is None:
        raise NotImplementedError
    if has_crypto and isinstance(hash_alg, type) and issubclass(hash_alg, hashes.HashAlgorithm):
        digest = hashes.Hash(hash_alg(), backend=default_backend())
        digest.update(bytestr)
        return bytes(digest.finalize())
    else:
        return bytes(hash_alg(bytestr).digest())