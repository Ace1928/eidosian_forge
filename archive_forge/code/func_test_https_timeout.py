import datetime
import json
import logging
import os.path
import shutil
import ssl
import sys
import tempfile
import warnings
from test import (
import mock
import pytest
import trustme
import urllib3.util as util
from dummyserver.server import (
from dummyserver.testcase import HTTPSDummyServerTestCase
from urllib3 import HTTPSConnectionPool
from urllib3.connection import RECENT_DATE, VerifiedHTTPSConnection
from urllib3.exceptions import (
from urllib3.packages import six
from urllib3.util.timeout import Timeout
from .. import has_alpn
@requires_network
def test_https_timeout(self):
    timeout = Timeout(total=None, connect=SHORT_TIMEOUT)
    with HTTPSConnectionPool(TARPIT_HOST, self.port, timeout=timeout, retries=False, cert_reqs='CERT_REQUIRED') as https_pool:
        with pytest.raises(ConnectTimeoutError):
            https_pool.request('GET', '/')
    timeout = Timeout(read=0.01)
    with HTTPSConnectionPool(self.host, self.port, timeout=timeout, retries=False, cert_reqs='CERT_REQUIRED') as https_pool:
        https_pool.ca_certs = DEFAULT_CA
        https_pool.assert_fingerprint = '72:8B:55:4C:9A:FC:1E:88:A1:1C:AD:1B:B2:E7:CC:3E:DB:C8:F9:8A'
    timeout = Timeout(total=None)
    with HTTPSConnectionPool(self.host, self.port, timeout=timeout, cert_reqs='CERT_NONE') as https_pool:
        https_pool.request('GET', '/')