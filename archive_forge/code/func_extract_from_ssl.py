import os
import platform
import socket
import ssl
import typing
import _ssl  # type: ignore[import]
from ._ssl_constants import (
def extract_from_ssl() -> None:
    """Restores the :class:`ssl.SSLContext` class to its original state"""
    setattr(ssl, 'SSLContext', _original_SSLContext)
    try:
        import pip._vendor.urllib3.util.ssl_ as urllib3_ssl
        urllib3_ssl.SSLContext = _original_SSLContext
    except ImportError:
        pass