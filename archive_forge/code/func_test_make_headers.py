import hashlib
import io
import logging
import socket
import ssl
import warnings
from itertools import chain
from test import notBrotlipy, onlyBrotlipy, onlyPy2, onlyPy3
import pytest
from mock import Mock, patch
from urllib3 import add_stderr_logger, disable_warnings, util
from urllib3.exceptions import (
from urllib3.packages import six
from urllib3.poolmanager import ProxyConfig
from urllib3.util import is_fp_closed
from urllib3.util.connection import _has_ipv6, allowed_gai_family, create_connection
from urllib3.util.proxy import connection_requires_http_tunnel, create_proxy_ssl_context
from urllib3.util.request import _FAILEDTELL, make_headers, rewind_body
from urllib3.util.response import assert_header_parsing
from urllib3.util.ssl_ import (
from urllib3.util.timeout import Timeout
from urllib3.util.url import Url, get_host, parse_url, split_first
from . import clear_warnings
@pytest.mark.parametrize('kwargs, expected', [pytest.param({'accept_encoding': True}, {'accept-encoding': 'gzip,deflate,br'}, marks=onlyBrotlipy()), pytest.param({'accept_encoding': True}, {'accept-encoding': 'gzip,deflate'}, marks=notBrotlipy()), ({'accept_encoding': 'foo,bar'}, {'accept-encoding': 'foo,bar'}), ({'accept_encoding': ['foo', 'bar']}, {'accept-encoding': 'foo,bar'}), pytest.param({'accept_encoding': True, 'user_agent': 'banana'}, {'accept-encoding': 'gzip,deflate,br', 'user-agent': 'banana'}, marks=onlyBrotlipy()), pytest.param({'accept_encoding': True, 'user_agent': 'banana'}, {'accept-encoding': 'gzip,deflate', 'user-agent': 'banana'}, marks=notBrotlipy()), ({'user_agent': 'banana'}, {'user-agent': 'banana'}), ({'keep_alive': True}, {'connection': 'keep-alive'}), ({'basic_auth': 'foo:bar'}, {'authorization': 'Basic Zm9vOmJhcg=='}), ({'proxy_basic_auth': 'foo:bar'}, {'proxy-authorization': 'Basic Zm9vOmJhcg=='}), ({'disable_cache': True}, {'cache-control': 'no-cache'})])
def test_make_headers(self, kwargs, expected):
    assert make_headers(**kwargs) == expected