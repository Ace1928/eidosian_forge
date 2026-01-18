from __future__ import print_function
import base64
import hashlib
import os
from cStringIO import StringIO
from M2Crypto import BIO, EVP, RSA, X509, m2
def key_info(pkey, cert, key_info_template):
    """Convert private key (PEM) to XML Signature format (RSAKeyValue/X509Data)"""
    exponent = base64.b64encode(pkey.e[4:])
    modulus = m2.bn_to_hex(m2.mpi_to_bn(pkey.n)).decode('hex').encode('base64')
    x509 = x509_parse_cert(cert) if cert else None
    return key_info_template % {'modulus': modulus, 'exponent': exponent, 'issuer_name': x509.get_issuer().as_text() if x509 else '', 'serial_number': x509.get_serial_number() if x509 else ''}