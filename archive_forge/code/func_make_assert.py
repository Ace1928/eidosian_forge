import os
import sys
import warnings
from typing import Any, Callable, NoReturn, Type, Union
from cryptography.hazmat.bindings.openssl.binding import Binding
def make_assert(error: Type[Exception]) -> Callable[[bool], Any]:
    """
    Create an assert function that uses :func:`exception_from_error_queue` to
    raise an exception wrapped by *error*.
    """

    def openssl_assert(ok: bool) -> None:
        """
        If *ok* is not True, retrieve the error from OpenSSL and raise it.
        """
        if ok is not True:
            exception_from_error_queue(error)
    return openssl_assert