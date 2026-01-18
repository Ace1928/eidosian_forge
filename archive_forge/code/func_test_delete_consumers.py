from unittest import mock
import fixtures
from urllib import parse as urlparse
import uuid
from testtools import matchers
from keystoneclient import session
from keystoneclient.tests.unit.v3 import client_fixtures
from keystoneclient.tests.unit.v3 import utils
from keystoneclient import utils as client_utils
from keystoneclient.v3.contrib.oauth1 import access_tokens
from keystoneclient.v3.contrib.oauth1 import auth
from keystoneclient.v3.contrib.oauth1 import consumers
from keystoneclient.v3.contrib.oauth1 import request_tokens
def test_delete_consumers(self):
    get_mock = self._mock_request_method(method='delete')
    _, resp = self.mgr.delete('admin')
    self.assertEqual(resp.request_ids[0], self.TEST_REQUEST_ID)
    get_mock.assert_called_once_with('/OS-OAUTH1/consumers/admin')