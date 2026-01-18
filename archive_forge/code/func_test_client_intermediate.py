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
def test_client_intermediate(self):
    """Check that certificate chains work well with client certs

        We generate an intermediate CA from the root CA, and issue a client certificate
        from that intermediate CA. Since the server only knows about the root CA, we
        need to send it the certificate *and* the intermediate CA, so that it can check
        the whole chain.
        """
    with HTTPSConnectionPool(self.host, self.port, key_file=os.path.join(self.certs_dir, CLIENT_INTERMEDIATE_KEY), cert_file=os.path.join(self.certs_dir, CLIENT_INTERMEDIATE_PEM), ca_certs=DEFAULT_CA) as https_pool:
        r = https_pool.request('GET', '/certificate')
        subject = json.loads(r.data.decode('utf-8'))
        assert subject['organizationalUnitName'].startswith('Testing cert')