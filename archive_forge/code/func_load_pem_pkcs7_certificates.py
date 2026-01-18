from __future__ import annotations
import collections
import contextlib
import itertools
import typing
from contextlib import contextmanager
from cryptography import utils, x509
from cryptography.exceptions import UnsupportedAlgorithm, _Reasons
from cryptography.hazmat.backends.openssl import aead
from cryptography.hazmat.backends.openssl.ciphers import _CipherContext
from cryptography.hazmat.backends.openssl.cmac import _CMACContext
from cryptography.hazmat.backends.openssl.ec import (
from cryptography.hazmat.backends.openssl.rsa import (
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
from cryptography.hazmat.bindings.openssl import binding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives._asymmetric import AsymmetricPadding
from cryptography.hazmat.primitives.asymmetric import (
from cryptography.hazmat.primitives.asymmetric.padding import (
from cryptography.hazmat.primitives.asymmetric.types import (
from cryptography.hazmat.primitives.ciphers import (
from cryptography.hazmat.primitives.ciphers.algorithms import (
from cryptography.hazmat.primitives.ciphers.modes import (
from cryptography.hazmat.primitives.serialization import ssh
from cryptography.hazmat.primitives.serialization.pkcs12 import (
def load_pem_pkcs7_certificates(self, data: bytes) -> typing.List[x509.Certificate]:
    utils._check_bytes('data', data)
    bio = self._bytes_to_bio(data)
    p7 = self._lib.PEM_read_bio_PKCS7(bio.bio, self._ffi.NULL, self._ffi.NULL, self._ffi.NULL)
    if p7 == self._ffi.NULL:
        self._consume_errors()
        raise ValueError('Unable to parse PKCS7 data')
    p7 = self._ffi.gc(p7, self._lib.PKCS7_free)
    return self._load_pkcs7_certificates(p7)