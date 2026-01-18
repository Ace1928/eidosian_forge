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
def get_default_algorithms() -> dict[str, Algorithm]:
    """
    Returns the algorithms that are implemented by the library.
    """
    default_algorithms = {'none': NoneAlgorithm(), 'HS256': HMACAlgorithm(HMACAlgorithm.SHA256), 'HS384': HMACAlgorithm(HMACAlgorithm.SHA384), 'HS512': HMACAlgorithm(HMACAlgorithm.SHA512)}
    if has_crypto:
        default_algorithms.update({'RS256': RSAAlgorithm(RSAAlgorithm.SHA256), 'RS384': RSAAlgorithm(RSAAlgorithm.SHA384), 'RS512': RSAAlgorithm(RSAAlgorithm.SHA512), 'ES256': ECAlgorithm(ECAlgorithm.SHA256), 'ES256K': ECAlgorithm(ECAlgorithm.SHA256), 'ES384': ECAlgorithm(ECAlgorithm.SHA384), 'ES521': ECAlgorithm(ECAlgorithm.SHA512), 'ES512': ECAlgorithm(ECAlgorithm.SHA512), 'PS256': RSAPSSAlgorithm(RSAPSSAlgorithm.SHA256), 'PS384': RSAPSSAlgorithm(RSAPSSAlgorithm.SHA384), 'PS512': RSAPSSAlgorithm(RSAPSSAlgorithm.SHA512), 'EdDSA': OKPAlgorithm()})
    return default_algorithms