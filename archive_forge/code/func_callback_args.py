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
@property
def callback_args(self) -> Any:
    if self._passphrase is None:
        return _ffi.NULL
    elif isinstance(self._passphrase, bytes) or callable(self._passphrase):
        return _ffi.NULL
    else:
        raise TypeError('Last argument must be a byte string or a callable.')