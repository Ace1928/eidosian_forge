import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_empty_endpoint_groups_in_project(self):
    """Test when no endpoint groups associated with the project."""
    url = '/OS-EP-FILTER/projects/%(project_id)s/endpoint_groups' % {'project_id': self.project_id}
    response = self.get(url, expected_status=http.client.OK)
    self.assertEqual(0, len(response.result['endpoint_groups']))
    self.head(url, expected_status=http.client.OK)