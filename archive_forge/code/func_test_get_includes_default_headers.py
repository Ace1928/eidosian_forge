from unittest import mock
from keystoneauth1 import session
from requests_mock.contrib import fixture
import testtools
from barbicanclient import client
from barbicanclient import exceptions
from barbicanclient.exceptions import UnsupportedVersion
from barbicanclient.tests.utils import get_server_supported_versions
from barbicanclient.tests.utils import get_version_endpoint
from barbicanclient.tests.utils import mock_session
from barbicanclient.tests.utils import mock_session_get
from barbicanclient.tests.utils import mock_session_get_endpoint
def test_get_includes_default_headers(self):
    self.httpclient._default_headers = {'Test-Default-Header': 'test'}
    self.httpclient.get(self.href)
    self.assertEqual('test', self.get_mock.last_request.headers['Test-Default-Header'])