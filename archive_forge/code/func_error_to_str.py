from __future__ import annotations
from enum import Enum
from typing import Any
from _argon2_cffi_bindings import ffi, lib
from ._typing import Literal
from .exceptions import HashingError, VerificationError, VerifyMismatchError
def error_to_str(error: int) -> str:
    """
    Convert an Argon2 error code into a native string.

    :param int error: An Argon2 error code as returned by :func:`core`.

    :rtype: str

    .. versionadded:: 16.0.0
    """
    return ffi.string(lib.argon2_error_message(error)).decode('ascii')