import logging
import warnings
from rsa._compat import range
import rsa.prime
import rsa.pem
import rsa.common
import rsa.randnum
import rsa.core
@classmethod
def load_pkcs1_openssl_pem(cls, keyfile):
    """Loads a PKCS#1.5 PEM-encoded public key file from OpenSSL.

        These files can be recognised in that they start with BEGIN PUBLIC KEY
        rather than BEGIN RSA PUBLIC KEY.

        The contents of the file before the "-----BEGIN PUBLIC KEY-----" and
        after the "-----END PUBLIC KEY-----" lines is ignored.

        :param keyfile: contents of a PEM-encoded file that contains the public
            key, from OpenSSL.
        :type keyfile: bytes
        :return: a PublicKey object
        """
    der = rsa.pem.load_pem(keyfile, 'PUBLIC KEY')
    return cls.load_pkcs1_openssl_der(der)