from __future__ import annotations
import abc
import datetime
import os
import typing
from cryptography import utils
from cryptography.hazmat.bindings._rust import x509 as rust_x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import (
from cryptography.hazmat.primitives.asymmetric.types import (
from cryptography.x509.extensions import (
from cryptography.x509.name import Name, _ASN1Type
from cryptography.x509.oid import ObjectIdentifier
def add_revoked_certificate(self, revoked_certificate: RevokedCertificate) -> CertificateRevocationListBuilder:
    """
        Adds a revoked certificate to the CRL.
        """
    if not isinstance(revoked_certificate, RevokedCertificate):
        raise TypeError('Must be an instance of RevokedCertificate')
    return CertificateRevocationListBuilder(self._issuer_name, self._last_update, self._next_update, self._extensions, self._revoked_certificates + [revoked_certificate])