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
def test_assert_fingerprint_md5(self):
    with HTTPSConnectionPool('localhost', self.port, cert_reqs='CERT_REQUIRED', ca_certs=DEFAULT_CA) as https_pool:
        https_pool.assert_fingerprint = '55:39:BF:70:05:12:43:FA:1F:D1:BF:4E:E8:1B:07:1D'
        https_pool.request('GET', '/')