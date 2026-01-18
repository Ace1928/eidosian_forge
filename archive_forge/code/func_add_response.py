from __future__ import annotations
import abc
import datetime
import typing
from cryptography import utils, x509
from cryptography.hazmat.bindings._rust import ocsp
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.types import (
from cryptography.x509.base import (
def add_response(self, cert: x509.Certificate, issuer: x509.Certificate, algorithm: hashes.HashAlgorithm, cert_status: OCSPCertStatus, this_update: datetime.datetime, next_update: typing.Optional[datetime.datetime], revocation_time: typing.Optional[datetime.datetime], revocation_reason: typing.Optional[x509.ReasonFlags]) -> OCSPResponseBuilder:
    if self._response is not None:
        raise ValueError('Only one response per OCSPResponse.')
    singleresp = _SingleResponse(cert, issuer, algorithm, cert_status, this_update, next_update, revocation_time, revocation_reason)
    return OCSPResponseBuilder(singleresp, self._responder_id, self._certs, self._extensions)