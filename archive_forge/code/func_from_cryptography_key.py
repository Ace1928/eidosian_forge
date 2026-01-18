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
def from_cryptography_key(cls, crypto_key: _Key) -> 'PKey':
    """
        Construct based on a ``cryptography`` *crypto_key*.

        :param crypto_key: A ``cryptography`` key.
        :type crypto_key: One of ``cryptography``'s `key interfaces`_.

        :rtype: PKey

        .. versionadded:: 16.1.0
        """
    if not isinstance(crypto_key, (rsa.RSAPublicKey, rsa.RSAPrivateKey, dsa.DSAPublicKey, dsa.DSAPrivateKey, ec.EllipticCurvePrivateKey, ed25519.Ed25519PrivateKey, ed448.Ed448PrivateKey)):
        raise TypeError('Unsupported key type')
    from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat
    if isinstance(crypto_key, (rsa.RSAPublicKey, dsa.DSAPublicKey)):
        return load_publickey(FILETYPE_ASN1, crypto_key.public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo))
    else:
        der = crypto_key.private_bytes(Encoding.DER, PrivateFormat.PKCS8, NoEncryption())
        return load_privatekey(FILETYPE_ASN1, der)