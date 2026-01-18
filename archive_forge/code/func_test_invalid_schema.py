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
@pytest.mark.parametrize('url, error_msg', [('127.0.0.1', 'Proxy URL had no scheme, should start with http:// or https://'), ('localhost:8080', 'Proxy URL had no scheme, should start with http:// or https://'), ('ftp://google.com', 'Proxy URL had unsupported scheme ftp, should use http:// or https://')])
def test_invalid_schema(self, url, error_msg):
    with pytest.raises(ProxySchemeUnknown, match=error_msg):
        proxy_from_url(url)