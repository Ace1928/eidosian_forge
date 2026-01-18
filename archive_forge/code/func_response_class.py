from __future__ import (absolute_import, division, print_function)
import socket
from ansible.module_utils.six import PY2
from .basehttpadapter import BaseHTTPAdapter
from .. import constants
from .._import_helper import HTTPAdapter, urllib3, urllib3_connection
def response_class(self, sock, *args, **kwargs):
    if PY2:
        kwargs['buffering'] = not self.disable_buffering
    return super(UnixHTTPConnection, self).response_class(sock, *args, **kwargs)