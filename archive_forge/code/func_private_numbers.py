from __future__ import annotations
import typing
from cryptography.exceptions import (
from cryptography.hazmat.backends.openssl.utils import (
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
def private_numbers(self) -> ec.EllipticCurvePrivateNumbers:
    bn = self._backend._lib.EC_KEY_get0_private_key(self._ec_key)
    private_value = self._backend._bn_to_int(bn)
    return ec.EllipticCurvePrivateNumbers(private_value=private_value, public_numbers=self.public_key().public_numbers())