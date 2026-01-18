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
def load_privatekey(type: int, buffer: Union[str, bytes], passphrase: Optional[PassphraseCallableT]=None) -> PKey:
    """
    Load a private key (PKey) from the string *buffer* encoded with the type
    *type*.

    :param type: The file type (one of FILETYPE_PEM, FILETYPE_ASN1)
    :param buffer: The buffer the key is stored in
    :param passphrase: (optional) if encrypted PEM format, this can be
                       either the passphrase to use, or a callback for
                       providing the passphrase.

    :return: The PKey object
    """
    if isinstance(buffer, str):
        buffer = buffer.encode('ascii')
    bio = _new_mem_buf(buffer)
    helper = _PassphraseHelper(type, passphrase)
    if type == FILETYPE_PEM:
        evp_pkey = _lib.PEM_read_bio_PrivateKey(bio, _ffi.NULL, helper.callback, helper.callback_args)
        helper.raise_if_problem()
    elif type == FILETYPE_ASN1:
        evp_pkey = _lib.d2i_PrivateKey_bio(bio, _ffi.NULL)
    else:
        raise ValueError('type argument must be FILETYPE_PEM or FILETYPE_ASN1')
    if evp_pkey == _ffi.NULL:
        _raise_current_error()
    pkey = PKey.__new__(PKey)
    pkey._pkey = _ffi.gc(evp_pkey, _lib.EVP_PKEY_free)
    return pkey