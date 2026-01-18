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
def from_nid(cls, lib: Any, nid: int) -> '_EllipticCurve':
    """
        Instantiate a new :py:class:`_EllipticCurve` associated with the given
        OpenSSL NID.

        :param lib: The OpenSSL library binding object.

        :param nid: The OpenSSL NID the resulting curve object will represent.
            This must be a curve NID (and not, for example, a hash NID) or
            subsequent operations will fail in unpredictable ways.
        :type nid: :py:class:`int`

        :return: The curve object.
        """
    return cls(lib, nid, _ffi.string(lib.OBJ_nid2sn(nid)).decode('ascii'))