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
@requires_ssl_context_keyfile_password
def test_client_encrypted_key_requires_password(self):
    with HTTPSConnectionPool(self.host, self.port, key_file=os.path.join(self.certs_dir, PASSWORD_CLIENT_KEYFILE), cert_file=os.path.join(self.certs_dir, CLIENT_CERT), key_password=None) as https_pool:
        with pytest.raises(MaxRetryError) as e:
            https_pool.request('GET', '/certificate')
        assert 'password is required' in str(e.value)
        assert isinstance(e.value.reason, SSLError)