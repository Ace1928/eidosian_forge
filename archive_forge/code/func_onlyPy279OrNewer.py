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
def onlyPy279OrNewer(test):
    """Skips this test unless you are on Python 2.7.9 or later."""

    @six.wraps(test)
    def wrapper(*args, **kwargs):
        msg = '{name} requires Python 2.7.9+ to run'.format(name=test.__name__)
        if sys.version_info < (2, 7, 9):
            pytest.skip(msg)
        return test(*args, **kwargs)
    return wrapper