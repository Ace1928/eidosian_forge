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
@onlyPy279OrNewer
@notSecureTransport
@notOpenSSL098
def test_ca_dir_verified(self, tmpdir):
    shutil.copyfile(DEFAULT_CA, str(tmpdir / '81deb5f7.0'))
    with HTTPSConnectionPool(self.host, self.port, cert_reqs='CERT_REQUIRED', ca_cert_dir=str(tmpdir)) as https_pool:
        conn = https_pool._new_conn()
        assert conn.__class__ == VerifiedHTTPSConnection
        with warnings.catch_warnings(record=True) as w:
            r = https_pool.request('GET', '/')
            assert r.status == 200
        if self.tls_protocol_deprecated():
            w = [x for x in w if x.category != DeprecationWarning]
        assert w == []