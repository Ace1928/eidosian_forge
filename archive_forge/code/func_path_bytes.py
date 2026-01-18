import os
import sys
import warnings
from typing import Any, Callable, NoReturn, Type, Union
from cryptography.hazmat.bindings.openssl.binding import Binding
def path_bytes(s: StrOrBytesPath) -> bytes:
    """
    Convert a Python path to a :py:class:`bytes` for the path which can be
    passed into an OpenSSL API accepting a filename.

    :param s: A path (valid for os.fspath).

    :return: An instance of :py:class:`bytes`.
    """
    b = os.fspath(s)
    if isinstance(b, str):
        return b.encode(sys.getfilesystemencoding())
    else:
        return b