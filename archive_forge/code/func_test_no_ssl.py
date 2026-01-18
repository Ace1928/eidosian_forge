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
def test_no_ssl(self):
    with HTTPSConnectionPool(self.host, self.port) as pool:
        pool.ConnectionCls = None
        with pytest.raises(SSLError):
            pool._new_conn()
        with pytest.raises(MaxRetryError) as cm:
            pool.request('GET', '/', retries=0)
        assert isinstance(cm.value.reason, SSLError)