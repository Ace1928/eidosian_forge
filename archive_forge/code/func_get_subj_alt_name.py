from __future__ import absolute_import
import OpenSSL.SSL
from cryptography import x509
from cryptography.hazmat.backends.openssl import backend as openssl_backend
from cryptography.hazmat.backends.openssl.x509 import _Certificate
from io import BytesIO
from socket import error as SocketError
from socket import timeout
import logging
import ssl
import sys
from .. import util
from ..packages import six
from ..util.ssl_ import PROTOCOL_TLS_CLIENT
def get_subj_alt_name(peer_cert):
    """
    Given an PyOpenSSL certificate, provides all the subject alternative names.
    """
    if hasattr(peer_cert, 'to_cryptography'):
        cert = peer_cert.to_cryptography()
    else:
        cert = _Certificate(openssl_backend, peer_cert._x509)
    try:
        ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except x509.ExtensionNotFound:
        return []
    except (x509.DuplicateExtension, UnsupportedExtension, x509.UnsupportedGeneralNameType, UnicodeError) as e:
        log.warning('A problem was encountered with the certificate that prevented urllib3 from finding the SubjectAlternativeName field. This can affect certificate validation. The error was %s', e)
        return []
    names = [('DNS', name) for name in map(_dnsname_to_stdlib, ext.get_values_for_type(x509.DNSName)) if name is not None]
    names.extend((('IP Address', str(name)) for name in ext.get_values_for_type(x509.IPAddress)))
    return names