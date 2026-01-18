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
def notPyPy2(test):
    """Skips this test on PyPy2"""

    @six.wraps(test)
    def wrapper(*args, **kwargs):
        msg = '{} fails with PyPy 2 dues to funcsigs bugs'.format(test.__name__)
        if platform.python_implementation() == 'PyPy' and sys.version_info[0] == 2:
            pytest.xfail(msg)
        return test(*args, **kwargs)
    return wrapper