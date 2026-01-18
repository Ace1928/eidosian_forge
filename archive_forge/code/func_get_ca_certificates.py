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
def get_ca_certificates(self) -> Optional[Tuple[X509, ...]]:
    """
        Get the CA certificates in the PKCS #12 structure.

        :return: A tuple with the CA certificates in the chain, or
            :py:const:`None` if there are none.
        :rtype: :py:class:`tuple` of :py:class:`X509` or :py:const:`None`
        """
    if self._cacerts is not None:
        return tuple(self._cacerts)
    return None