import io
import json
import logging
import os
import platform
import socket
import sys
import time
import warnings
from test import LONG_TIMEOUT, SHORT_TIMEOUT, onlyPy2
from threading import Event
import mock
import pytest
import six
from dummyserver.server import HAS_IPV6_AND_DNS, NoIPv6Warning
from dummyserver.testcase import HTTPDummyServerTestCase, SocketDummyServerTestCase
from urllib3 import HTTPConnectionPool, encode_multipart_formdata
from urllib3._collections import HTTPHeaderDict
from urllib3.connection import _get_default_user_agent
from urllib3.exceptions import (
from urllib3.packages.six import b, u
from urllib3.packages.six.moves.urllib.parse import urlencode
from urllib3.util import SKIP_HEADER, SKIPPABLE_HEADERS
from urllib3.util.retry import RequestHistory, Retry
from urllib3.util.timeout import Timeout
from .. import INVALID_SOURCE_ADDRESSES, TARPIT_HOST, VALID_SOURCE_ADDRESSES
from ..port_helpers import find_unused_port
def test_cleanup_on_connection_error(self):
    """
        Test that connections are recycled to the pool on
        connection errors where no http response is received.
        """
    poolsize = 3
    with HTTPConnectionPool(self.host, self.port, maxsize=poolsize, block=True) as http:
        assert http.pool.qsize() == poolsize
        with pytest.raises(MaxRetryError):
            http.request('GET', '/redirect', fields={'target': '/'}, release_conn=False, retries=0)
        r = http.request('GET', '/redirect', fields={'target': '/'}, release_conn=False, retries=1)
        r.release_conn()
        assert http.pool.qsize() == http.pool.maxsize