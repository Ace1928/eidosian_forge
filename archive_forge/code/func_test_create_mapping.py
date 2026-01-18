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
def test_create_mapping(self):
    body = {'mapping': {'name': 'admin'}}
    self._mock_request_method(method='post', body=body)
    put_mock = self._mock_request_method(method='put', body=body)
    response = self.mgr.create(mapping_id='admin', description='fake')
    self.assertEqual(response.request_ids[0], self.TEST_REQUEST_ID)
    put_mock.assert_called_once_with('OS-FEDERATION/mappings/admin', body={'mapping': {'description': 'fake'}})