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
def test_lazy_load_twice(self):
    with HTTPConnectionPool(self.host, self.port, block=True, maxsize=1, timeout=2) as pool:
        payload_size = 1024 * 2
        first_chunk = 512
        boundary = 'foo'
        req_data = {'count': 'a' * payload_size}
        resp_data = encode_multipart_formdata(req_data, boundary=boundary)[0]
        req2_data = {'count': 'b' * payload_size}
        resp2_data = encode_multipart_formdata(req2_data, boundary=boundary)[0]
        r1 = pool.request('POST', '/echo', fields=req_data, multipart_boundary=boundary, preload_content=False)
        assert r1.read(first_chunk) == resp_data[:first_chunk]
        try:
            r2 = pool.request('POST', '/echo', fields=req2_data, multipart_boundary=boundary, preload_content=False, pool_timeout=0.001)
            assert r2.read(first_chunk) == resp2_data[:first_chunk]
            assert r1.read() == resp_data[first_chunk:]
            assert r2.read() == resp2_data[first_chunk:]
            assert pool.num_requests == 2
        except EmptyPoolError:
            assert r1.read() == resp_data[first_chunk:]
            assert pool.num_requests == 1
        assert pool.num_connections == 1