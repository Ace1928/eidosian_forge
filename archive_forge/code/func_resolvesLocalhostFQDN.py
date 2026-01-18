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
def resolvesLocalhostFQDN(test):
    """Test requires successful resolving of 'localhost.'"""

    @six.wraps(test)
    def wrapper(*args, **kwargs):
        if not RESOLVES_LOCALHOST_FQDN:
            pytest.skip("Can't resolve localhost.")
        return test(*args, **kwargs)
    return wrapper