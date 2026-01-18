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
def test_context_combines_with_ca_certs(self):
    ctx = util.ssl_.create_urllib3_context(cert_reqs=ssl.CERT_REQUIRED)
    with HTTPSConnectionPool(self.host, self.port, ca_certs=DEFAULT_CA, ssl_context=ctx) as https_pool:
        conn = https_pool._new_conn()
        assert conn.__class__ == VerifiedHTTPSConnection
        with mock.patch('warnings.warn') as warn:
            r = https_pool.request('GET', '/')
            assert r.status == 200
            if sys.version_info >= (2, 7, 9) or util.IS_PYOPENSSL or util.IS_SECURETRANSPORT:
                assert not warn.called, warn.call_args_list
            else:
                assert warn.called
                if util.HAS_SNI:
                    call = warn.call_args_list[0]
                else:
                    call = warn.call_args_list[1]
                error = call[0][1]
                assert error == InsecurePlatformWarning