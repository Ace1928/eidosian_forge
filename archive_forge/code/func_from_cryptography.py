import calendar
import datetime
import functools
from base64 import b16encode
from functools import partial
from os import PathLike
from typing import (
from cryptography import utils, x509
from cryptography.hazmat.primitives.asymmetric import (
from OpenSSL._util import (
@classmethod
def from_cryptography(cls, crypto_crl: x509.CertificateRevocationList) -> 'CRL':
    """
        Construct based on a ``cryptography`` *crypto_crl*.

        :param crypto_crl: A ``cryptography`` certificate revocation list
        :type crypto_crl: ``cryptography.x509.CertificateRevocationList``

        :rtype: CRL

        .. versionadded:: 17.1.0
        """
    if not isinstance(crypto_crl, x509.CertificateRevocationList):
        raise TypeError('Must be a certificate revocation list')
    from cryptography.hazmat.primitives.serialization import Encoding
    der = crypto_crl.public_bytes(Encoding.DER)
    return load_crl(FILETYPE_ASN1, der)