import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_list_endpoint_groups(self):
    """GET & HEAD /OS-EP-FILTER/endpoint_groups."""
    endpoint_group_id = self._create_valid_endpoint_group(self.DEFAULT_ENDPOINT_GROUP_URL, self.DEFAULT_ENDPOINT_GROUP_BODY)
    url = '/OS-EP-FILTER/endpoint_groups'
    r = self.get(url, expected_status=http.client.OK)
    self.assertNotEmpty(r.result['endpoint_groups'])
    self.assertEqual(endpoint_group_id, r.result['endpoint_groups'][0].get('id'))
    self.head(url, expected_status=http.client.OK)