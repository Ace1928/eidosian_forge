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
def test_client_no_intermediate(self):
    """Check that missing links in certificate chains indeed break

        The only difference with test_client_intermediate is that we don't send the
        intermediate CA to the server, only the client cert.
        """
    with HTTPSConnectionPool(self.host, self.port, cert_file=os.path.join(self.certs_dir, CLIENT_NO_INTERMEDIATE_PEM), key_file=os.path.join(self.certs_dir, CLIENT_INTERMEDIATE_KEY), ca_certs=DEFAULT_CA) as https_pool:
        with pytest.raises((SSLError, ProtocolError)):
            https_pool.request('GET', '/certificate', retries=False)