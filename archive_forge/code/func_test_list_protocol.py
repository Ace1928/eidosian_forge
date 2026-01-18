import copy
import fixtures
import uuid
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1.identity import v3
from keystoneauth1 import session
from keystoneauth1.tests.unit import k2k_fixtures
from testtools import matchers
from keystoneclient import access
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import client
from keystoneclient.v3.contrib.federation import base
from keystoneclient.v3.contrib.federation import identity_providers
from keystoneclient.v3.contrib.federation import mappings
from keystoneclient.v3.contrib.federation import protocols
from keystoneclient.v3.contrib.federation import service_providers
from keystoneclient.v3 import domains
from keystoneclient.v3 import projects
def test_list_protocol(self):
    body = {'protocols': [{'name': 'admin'}]}
    get_mock = self._mock_request_method(method='get', body=body)
    response = self.mgr.list('identity_provider')
    self.assertEqual(response.request_ids[0], self.TEST_REQUEST_ID)
    get_mock.assert_called_once_with('OS-FEDERATION/identity_providers/identity_provider/protocols?')