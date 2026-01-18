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
def test_api_version_is_appended_to_endpoint(self):
    c = client.Client(session=self.session, endpoint=self.endpoint, project_id=self.project_id)
    self.assertEqual('http://localhost:9311/v1/', c.client.endpoint_override)