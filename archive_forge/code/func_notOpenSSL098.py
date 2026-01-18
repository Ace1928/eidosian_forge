import errno
import logging
import os
import platform
import socket
import ssl
import sys
import warnings
import pytest
from urllib3 import util
from urllib3.exceptions import HTTPWarning
from urllib3.packages import six
from urllib3.util import ssl_
def notOpenSSL098(test):
    """Skips this test for Python 3.5 macOS python.org distribution"""

    @six.wraps(test)
    def wrapper(*args, **kwargs):
        is_stdlib_ssl = not ssl_.IS_SECURETRANSPORT and (not ssl_.IS_PYOPENSSL)
        if is_stdlib_ssl and ssl.OPENSSL_VERSION == 'OpenSSL 0.9.8zh 14 Jan 2016':
            pytest.xfail('{name} fails with OpenSSL 0.9.8zh'.format(name=test.__name__))
        return test(*args, **kwargs)
    return wrapper