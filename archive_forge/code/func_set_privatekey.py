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
def set_privatekey(self, pkey: PKey) -> None:
    """
        Set the certificate portion of the PKCS #12 structure.

        :param pkey: The new private key, or :py:const:`None` to unset it.
        :type pkey: :py:class:`PKey` or :py:const:`None`

        :return: ``None``
        """
    if not isinstance(pkey, PKey):
        raise TypeError('pkey must be a PKey instance')
    self._pkey = pkey