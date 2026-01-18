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
def set_notAfter(self, when: bytes) -> None:
    """
        Set the timestamp at which the certificate stops being valid.

        The timestamp is formatted as an ASN.1 TIME::

            YYYYMMDDhhmmssZ

        :param bytes when: A timestamp string.
        :return: ``None``
        """
    return self._set_boundary_time(_lib.X509_getm_notAfter, when)