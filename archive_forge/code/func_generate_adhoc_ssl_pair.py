from __future__ import annotations
import errno
import io
import os
import selectors
import socket
import socketserver
import sys
import typing as t
from datetime import datetime as dt
from datetime import timedelta
from datetime import timezone
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
from urllib.parse import unquote
from urllib.parse import urlsplit
from ._internal import _log
from ._internal import _wsgi_encoding_dance
from .exceptions import InternalServerError
from .urls import uri_to_iri
def generate_adhoc_ssl_pair(cn: str | None=None) -> tuple[Certificate, RSAPrivateKeyWithSerialization]:
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import rsa
    except ImportError:
        raise TypeError('Using ad-hoc certificates requires the cryptography library.') from None
    backend = default_backend()
    pkey = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=backend)
    if cn is None:
        cn = '*'
    subject = x509.Name([x509.NameAttribute(NameOID.ORGANIZATION_NAME, 'Dummy Certificate'), x509.NameAttribute(NameOID.COMMON_NAME, cn)])
    backend = default_backend()
    cert = x509.CertificateBuilder().subject_name(subject).issuer_name(subject).public_key(pkey.public_key()).serial_number(x509.random_serial_number()).not_valid_before(dt.now(timezone.utc)).not_valid_after(dt.now(timezone.utc) + timedelta(days=365)).add_extension(x509.ExtendedKeyUsage([x509.OID_SERVER_AUTH]), critical=False).add_extension(x509.SubjectAlternativeName([x509.DNSName(cn)]), critical=False).sign(pkey, hashes.SHA256(), backend)
    return (cert, pkey)