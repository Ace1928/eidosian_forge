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
def test_alpn_default(self):
    """Default ALPN protocols are sent by default."""
    if not has_alpn() or not has_alpn(ssl.SSLContext):
        pytest.skip('ALPN-support not available')
    with HTTPSConnectionPool(self.host, self.port, ca_certs=DEFAULT_CA) as pool:
        r = pool.request('GET', '/alpn_protocol', retries=0)
        assert r.status == 200
        assert r.data.decode('utf-8') == util.ALPN_PROTOCOLS[0]