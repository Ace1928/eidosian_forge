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
@pytest.mark.parametrize('kwargs, message', [({'total': -1}, 'less than'), ({'connect': 2, 'total': -1}, 'less than'), ({'read': -1}, 'less than'), ({'connect': False}, 'cannot be a boolean'), ({'read': True}, 'cannot be a boolean'), ({'connect': 0}, 'less than or equal'), ({'read': 'foo'}, 'int, float or None')])
def test_invalid_timeouts(self, kwargs, message):
    with pytest.raises(ValueError) as e:
        Timeout(**kwargs)
    assert message in str(e.value)