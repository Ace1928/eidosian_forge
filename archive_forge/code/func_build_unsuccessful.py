from __future__ import annotations
import abc
import datetime
import typing
from cryptography import utils, x509
from cryptography.hazmat.bindings._rust import ocsp
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.types import (
from cryptography.x509.base import (
@classmethod
def build_unsuccessful(cls, response_status: OCSPResponseStatus) -> OCSPResponse:
    if not isinstance(response_status, OCSPResponseStatus):
        raise TypeError('response_status must be an item from OCSPResponseStatus')
    if response_status is OCSPResponseStatus.SUCCESSFUL:
        raise ValueError('response_status cannot be SUCCESSFUL')
    return ocsp.create_ocsp_response(response_status, None, None, None)