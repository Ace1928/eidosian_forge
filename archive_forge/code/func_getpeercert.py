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
def getpeercert(self, binary_form=False):
    x509 = self.connection.get_peer_certificate()
    if not x509:
        return x509
    if binary_form:
        return OpenSSL.crypto.dump_certificate(OpenSSL.crypto.FILETYPE_ASN1, x509)
    return {'subject': ((('commonName', x509.get_subject().CN),),), 'subjectAltName': get_subj_alt_name(x509)}