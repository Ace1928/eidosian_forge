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
def get_components(self) -> List[Tuple[bytes, bytes]]:
    """
        Returns the components of this name, as a sequence of 2-tuples.

        :return: The components of this name.
        :rtype: :py:class:`list` of ``name, value`` tuples.
        """
    result = []
    for i in range(_lib.X509_NAME_entry_count(self._name)):
        ent = _lib.X509_NAME_get_entry(self._name, i)
        fname = _lib.X509_NAME_ENTRY_get_object(ent)
        fval = _lib.X509_NAME_ENTRY_get_data(ent)
        nid = _lib.OBJ_obj2nid(fname)
        name = _lib.OBJ_nid2sn(nid)
        value = _ffi.buffer(_lib.ASN1_STRING_get0_data(fval), _lib.ASN1_STRING_length(fval))[:]
        result.append((_ffi.string(name), value))
    return result