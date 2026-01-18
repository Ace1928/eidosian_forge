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
def test_conn_closed(self):
    block_event = Event()
    self.start_basic_handler(block_send=block_event, num=1)
    with HTTPConnectionPool(self.host, self.port, timeout=SHORT_TIMEOUT, retries=False) as pool:
        conn = pool._get_conn()
        pool._put_conn(conn)
        try:
            with pytest.raises(ReadTimeoutError):
                pool.urlopen('GET', '/')
            if conn.sock:
                with pytest.raises(socket.error):
                    conn.sock.recv(1024)
        finally:
            pool._put_conn(conn)
        block_event.set()