from __future__ import annotations
import binascii
import enum
import os
import re
import typing
import warnings
from base64 import encodebytes as _base64_encode
from dataclasses import dataclass
from cryptography import utils
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import (
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
from cryptography.hazmat.primitives.ciphers import (
from cryptography.hazmat.primitives.serialization import (
def load_private(self, data: memoryview, pubfields) -> typing.Tuple[ed25519.Ed25519PrivateKey, memoryview]:
    """Make Ed25519 private key from data."""
    (point,), data = self.get_public(data)
    keypair, data = _get_sshstr(data)
    secret = keypair[:32]
    point2 = keypair[32:]
    if point != point2 or (point,) != pubfields:
        raise ValueError('Corrupt data: ed25519 field mismatch')
    private_key = ed25519.Ed25519PrivateKey.from_private_bytes(secret)
    return (private_key, data)