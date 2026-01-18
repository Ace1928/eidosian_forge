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
def get_subject(self) -> X509Name:
    """
        Return the subject of this certificate.

        This creates a new :class:`X509Name` that wraps the underlying subject
        name field on the certificate. Modifying it will modify the underlying
        certificate, and will have the effect of modifying any other
        :class:`X509Name` that refers to this subject.

        :return: The subject of this certificate.
        :rtype: :class:`X509Name`
        """
    name = self._get_name(_lib.X509_get_subject_name)
    self._subject_invalidator.add(name)
    return name