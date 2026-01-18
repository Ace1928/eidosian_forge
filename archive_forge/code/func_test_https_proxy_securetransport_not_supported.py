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
@onlySecureTransport
@onlyPy3
def test_https_proxy_securetransport_not_supported(self):
    with proxy_from_url(self.https_proxy_url, ca_certs=DEFAULT_CA) as https:
        r = https.request('GET', '%s/' % self.http_url)
        assert r.status == 200
        with pytest.raises(ProxySchemeUnsupported) as excinfo:
            https.request('GET', '%s/' % self.https_url)
        assert "isn't available on non-native SSLContext" in str(excinfo.value)