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
def notSecureTransport(test):
    """Skips this test when SecureTransport is in use."""

    @six.wraps(test)
    def wrapper(*args, **kwargs):
        msg = '{name} does not run with SecureTransport'.format(name=test.__name__)
        if ssl_.IS_SECURETRANSPORT:
            pytest.skip(msg)
        return test(*args, **kwargs)
    return wrapper