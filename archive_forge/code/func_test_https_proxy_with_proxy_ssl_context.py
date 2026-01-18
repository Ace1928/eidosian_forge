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
@onlyPy3
def test_https_proxy_with_proxy_ssl_context(self):
    proxy_ssl_context = create_urllib3_context()
    proxy_ssl_context.load_verify_locations(DEFAULT_CA)
    with proxy_from_url(self.https_proxy_url, proxy_ssl_context=proxy_ssl_context, ca_certs=DEFAULT_CA) as https:
        r = https.request('GET', '%s/' % self.https_url)
        assert r.status == 200
        r = https.request('GET', '%s/' % self.http_url)
        assert r.status == 200