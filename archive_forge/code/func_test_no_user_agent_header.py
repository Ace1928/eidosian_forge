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
def test_no_user_agent_header(self):
    """ConnectionPool can suppress sending a user agent header"""
    custom_ua = "I'm not a web scraper, what are you talking about?"
    with HTTPConnectionPool(self.host, self.port) as pool:
        no_ua_headers = {'User-Agent': SKIP_HEADER}
        r = pool.request('GET', '/headers', headers=no_ua_headers)
        request_headers = json.loads(r.data.decode('utf8'))
        assert 'User-Agent' not in request_headers
        assert no_ua_headers['User-Agent'] == SKIP_HEADER
        pool.headers = no_ua_headers
        r = pool.request('GET', '/headers')
        request_headers = json.loads(r.data.decode('utf8'))
        assert 'User-Agent' not in request_headers
        assert no_ua_headers['User-Agent'] == SKIP_HEADER
        pool_headers = {'User-Agent': custom_ua}
        pool.headers = pool_headers
        r = pool.request('GET', '/headers', headers=no_ua_headers)
        request_headers = json.loads(r.data.decode('utf8'))
        assert 'User-Agent' not in request_headers
        assert no_ua_headers['User-Agent'] == SKIP_HEADER
        assert pool_headers.get('User-Agent') == custom_ua