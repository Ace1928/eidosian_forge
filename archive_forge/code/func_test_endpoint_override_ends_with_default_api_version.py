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
def test_endpoint_override_ends_with_default_api_version(self):
    c = client.Client(session=self.session, endpoint=self.endpoint, project_id=self.project_id)
    self.assertTrue(c.client.endpoint_override.rstrip('/').endswith(client._DEFAULT_API_VERSION))