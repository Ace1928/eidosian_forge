from __future__ import annotations
import json
import time
from typing import Any
from .algorithms import get_default_algorithms, has_crypto, requires_cryptography
from .exceptions import InvalidKeyError, PyJWKError, PyJWKSetError, PyJWTError
from .types import JWKDict
def get_jwk_set(self) -> PyJWKSet:
    return self.jwk_set