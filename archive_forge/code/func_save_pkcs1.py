import logging
import warnings
from rsa._compat import range
import rsa.prime
import rsa.pem
import rsa.common
import rsa.randnum
import rsa.core
def save_pkcs1(self, format='PEM'):
    """Saves the key in PKCS#1 DER or PEM format.

        :param format: the format to save; 'PEM' or 'DER'
        :type format: str
        :returns: the DER- or PEM-encoded key.
        :rtype: bytes
        """
    methods = {'PEM': self._save_pkcs1_pem, 'DER': self._save_pkcs1_der}
    method = self._assert_format_exists(format, methods)
    return method()