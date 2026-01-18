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
def set_ca_certificates(self, cacerts: Optional[Iterable[X509]]) -> None:
    """
        Replace or set the CA certificates within the PKCS12 object.

        :param cacerts: The new CA certificates, or :py:const:`None` to unset
            them.
        :type cacerts: An iterable of :py:class:`X509` or :py:const:`None`

        :return: ``None``
        """
    if cacerts is None:
        self._cacerts = None
    else:
        cacerts = list(cacerts)
        for cert in cacerts:
            if not isinstance(cert, X509):
                raise TypeError('iterable must only contain X509 instances')
        self._cacerts = cacerts