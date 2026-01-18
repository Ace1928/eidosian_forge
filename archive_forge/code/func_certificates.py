from __future__ import annotations
import abc
import datetime
import typing
from cryptography import utils, x509
from cryptography.hazmat.bindings._rust import ocsp
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.types import (
from cryptography.x509.base import (
def certificates(self, certs: typing.Iterable[x509.Certificate]) -> OCSPResponseBuilder:
    if self._certs is not None:
        raise ValueError('certificates may only be set once')
    certs = list(certs)
    if len(certs) == 0:
        raise ValueError('certs must not be an empty list')
    if not all((isinstance(x, x509.Certificate) for x in certs)):
        raise TypeError('certs must be a list of Certificates')
    return OCSPResponseBuilder(self._response, self._responder_id, certs, self._extensions)