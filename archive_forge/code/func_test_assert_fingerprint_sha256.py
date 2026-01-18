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
def test_assert_fingerprint_sha256(self):
    with HTTPSConnectionPool('localhost', self.port, cert_reqs='CERT_REQUIRED', ca_certs=DEFAULT_CA) as https_pool:
        https_pool.assert_fingerprint = 'E3:59:8E:69:FF:C5:9F:C7:88:87:44:58:22:7F:90:8D:D9:BC:12:C4:90:79:D5:DC:A8:5D:4F:60:40:1E:A6:D2'
        https_pool.request('GET', '/')