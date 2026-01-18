from unittest import mock
from novaclient import api_versions
from novaclient import exceptions as exc
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import versions
def test_get_endpoint_without_project_id(self):
    endpoint_type = 'v2.1'
    expected_endpoint = 'http://nova-api:8774/v2.1/'
    cs_2_1 = fakes.FakeClient(api_versions.APIVersion('2.0'), endpoint_type=endpoint_type)
    result = cs_2_1.versions.get_current()
    self.assert_request_id(result, fakes.FAKE_REQUEST_ID_LIST)
    self.assertEqual(result.manager.api.client.endpoint_type, endpoint_type, 'Check endpoint_type was set')
    cs_2_1.assert_called('GET', expected_endpoint)