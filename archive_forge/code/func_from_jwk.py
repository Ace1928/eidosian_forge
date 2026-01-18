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
@staticmethod
def from_jwk(jwk: str | JWKDict) -> AllowedOKPKeys:
    try:
        if isinstance(jwk, str):
            obj = json.loads(jwk)
        elif isinstance(jwk, dict):
            obj = jwk
        else:
            raise ValueError
    except ValueError:
        raise InvalidKeyError('Key is not valid JSON')
    if obj.get('kty') != 'OKP':
        raise InvalidKeyError('Not an Octet Key Pair')
    curve = obj.get('crv')
    if curve != 'Ed25519' and curve != 'Ed448':
        raise InvalidKeyError(f'Invalid curve: {curve}')
    if 'x' not in obj:
        raise InvalidKeyError('OKP should have "x" parameter')
    x = base64url_decode(obj.get('x'))
    try:
        if 'd' not in obj:
            if curve == 'Ed25519':
                return Ed25519PublicKey.from_public_bytes(x)
            return Ed448PublicKey.from_public_bytes(x)
        d = base64url_decode(obj.get('d'))
        if curve == 'Ed25519':
            return Ed25519PrivateKey.from_private_bytes(d)
        return Ed448PrivateKey.from_private_bytes(d)
    except ValueError as err:
        raise InvalidKeyError('Invalid key parameter') from err