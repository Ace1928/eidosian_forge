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
def requires_network(test):
    """Helps you skip tests that require the network"""

    def _is_unreachable_err(err):
        return getattr(err, 'errno', None) in (errno.ENETUNREACH, errno.EHOSTUNREACH)

    def _has_route():
        try:
            sock = socket.create_connection((TARPIT_HOST, 80), 0.0001)
            sock.close()
            return True
        except socket.timeout:
            return True
        except socket.error as e:
            if _is_unreachable_err(e):
                return False
            else:
                raise

    @six.wraps(test)
    def wrapper(*args, **kwargs):
        global _requires_network_has_route
        if _requires_network_has_route is None:
            _requires_network_has_route = _has_route()
        if _requires_network_has_route:
            return test(*args, **kwargs)
        else:
            msg = "Can't run {name} because the network is unreachable".format(name=test.__name__)
            pytest.skip(msg)
    return wrapper