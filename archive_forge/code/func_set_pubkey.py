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
def set_pubkey(self, pkey: PKey) -> None:
    """
        Set the public key of the certificate

        :param pkey: The public key
        :return: ``None``
        """
    set_result = _lib.NETSCAPE_SPKI_set_pubkey(self._spki, pkey._pkey)
    _openssl_assert(set_result == 1)