from __future__ import annotations
import abc
import typing
from cryptography import utils
from cryptography.hazmat._oid import ObjectIdentifier
from cryptography.hazmat.primitives import _serialization, hashes
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
def derive_private_key(private_value: int, curve: EllipticCurve, backend: typing.Any=None) -> EllipticCurvePrivateKey:
    from cryptography.hazmat.backends.openssl.backend import backend as ossl
    if not isinstance(private_value, int):
        raise TypeError('private_value must be an integer type.')
    if private_value <= 0:
        raise ValueError('private_value must be a positive integer.')
    if not isinstance(curve, EllipticCurve):
        raise TypeError('curve must provide the EllipticCurve interface.')
    return ossl.derive_elliptic_curve_private_key(private_value, curve)