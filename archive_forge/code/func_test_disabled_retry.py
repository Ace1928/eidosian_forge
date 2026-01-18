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
def test_disabled_retry(self):
    """Disabled retries should disable redirect handling."""
    with HTTPConnectionPool(self.host, self.port) as pool:
        r = pool.request('GET', '/redirect', fields={'target': '/'}, retries=False)
        assert r.status == 303
        r = pool.request('GET', '/redirect', fields={'target': '/'}, retries=Retry(redirect=False))
        assert r.status == 303
    with HTTPConnectionPool('thishostdoesnotexist.invalid', self.port, timeout=0.001) as pool:
        with pytest.raises(NewConnectionError):
            pool.request('GET', '/test', retries=False)