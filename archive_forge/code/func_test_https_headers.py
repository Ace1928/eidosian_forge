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
def test_https_headers(self):
    with proxy_from_url(self.https_proxy_url, headers={'Foo': 'bar'}, proxy_headers={'Hickory': 'dickory'}, ca_certs=DEFAULT_CA) as http:
        r = http.request_encode_url('GET', '%s/headers' % self.http_url)
        returned_headers = json.loads(r.data.decode())
        assert returned_headers.get('Foo') == 'bar'
        assert returned_headers.get('Hickory') == 'dickory'
        assert returned_headers.get('Host') == '%s:%s' % (self.http_host, self.http_port)
        r = http.request_encode_url('GET', '%s/headers' % self.http_url_alt)
        returned_headers = json.loads(r.data.decode())
        assert returned_headers.get('Foo') == 'bar'
        assert returned_headers.get('Hickory') == 'dickory'
        assert returned_headers.get('Host') == '%s:%s' % (self.http_host_alt, self.http_port)
        r = http.request_encode_body('GET', '%s/headers' % self.https_url, headers={'Baz': 'quux'})
        returned_headers = json.loads(r.data.decode())
        assert returned_headers.get('Foo') is None
        assert returned_headers.get('Baz') == 'quux'
        assert returned_headers.get('Hickory') is None
        assert returned_headers.get('Host') == '%s:%s' % (self.https_host, self.https_port)