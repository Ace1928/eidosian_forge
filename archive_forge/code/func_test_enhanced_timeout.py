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
def test_enhanced_timeout(self):
    with HTTPSConnectionPool(TARPIT_HOST, self.port, timeout=Timeout(connect=SHORT_TIMEOUT), retries=False, cert_reqs='CERT_REQUIRED') as https_pool:
        conn = https_pool._new_conn()
        try:
            with pytest.raises(ConnectTimeoutError):
                https_pool.request('GET', '/')
            with pytest.raises(ConnectTimeoutError):
                https_pool._make_request(conn, 'GET', '/')
        finally:
            conn.close()
    with HTTPSConnectionPool(TARPIT_HOST, self.port, timeout=Timeout(connect=LONG_TIMEOUT), retries=False, cert_reqs='CERT_REQUIRED') as https_pool:
        with pytest.raises(ConnectTimeoutError):
            https_pool.request('GET', '/', timeout=Timeout(connect=SHORT_TIMEOUT))
    with HTTPSConnectionPool(TARPIT_HOST, self.port, timeout=Timeout(total=None), retries=False, cert_reqs='CERT_REQUIRED') as https_pool:
        conn = https_pool._new_conn()
        try:
            with pytest.raises(ConnectTimeoutError):
                https_pool.request('GET', '/', timeout=Timeout(total=None, connect=SHORT_TIMEOUT))
        finally:
            conn.close()