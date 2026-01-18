import json
import os.path
import shutil
import socket
import ssl
import sys
import tempfile
import warnings
from test import (
import pytest
import trustme
from dummyserver.server import DEFAULT_CA, HAS_IPV6, get_unreachable_address
from dummyserver.testcase import HTTPDummyProxyTestCase, IPv6HTTPDummyProxyTestCase
from urllib3._collections import HTTPHeaderDict
from urllib3.connectionpool import VerifiedHTTPSConnection, connection_from_url
from urllib3.exceptions import (
from urllib3.poolmanager import ProxyManager, proxy_from_url
from urllib3.util import Timeout
from urllib3.util.ssl_ import create_urllib3_context
from .. import TARPIT_HOST, requires_network
def test_nagle_proxy(self):
    """Test that proxy connections do not have TCP_NODELAY turned on"""
    with ProxyManager(self.proxy_url) as http:
        hc2 = http.connection_from_host(self.http_host, self.http_port)
        conn = hc2._get_conn()
        try:
            hc2._make_request(conn, 'GET', '/')
            tcp_nodelay_setting = conn.sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)
            assert tcp_nodelay_setting == 0, 'Expected TCP_NODELAY for proxies to be set to zero, instead was %s' % tcp_nodelay_setting
        finally:
            conn.close()