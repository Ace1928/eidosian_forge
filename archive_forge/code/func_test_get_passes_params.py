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
def test_get_passes_params(self):
    params = {'test': 'test1'}
    self.httpclient.get(self.href, params=params)
    self.assertEqual(self.href, self.get_mock.last_request.url.split('?')[0])
    self.assertEqual(['test1'], self.get_mock.last_request.qs['test'])