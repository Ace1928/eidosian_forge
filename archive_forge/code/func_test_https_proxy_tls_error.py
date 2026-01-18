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
@requires_network
@pytest.mark.parametrize(['target_scheme', 'use_forwarding_for_https'], [('http', False), ('https', False), ('https', True)])
def test_https_proxy_tls_error(self, target_scheme, use_forwarding_for_https):
    _should_skip_https_in_https('https', target_scheme, use_forwarding_for_https)
    target_url = self.https_url if target_scheme == 'https' else self.http_url
    proxy_ctx = ssl.create_default_context()
    with proxy_from_url(self.https_proxy_url, proxy_ssl_context=proxy_ctx, use_forwarding_for_https=use_forwarding_for_https) as proxy:
        with pytest.raises(MaxRetryError) as e:
            proxy.request('GET', target_url)
        assert type(e.value.reason) == SSLError