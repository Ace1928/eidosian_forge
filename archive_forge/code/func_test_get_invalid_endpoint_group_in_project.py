import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_get_invalid_endpoint_group_in_project(self):
    """Test retrieving project endpoint group association."""
    endpoint_group_id = uuid.uuid4().hex
    project_id = uuid.uuid4().hex
    url = self._get_project_endpoint_group_url(endpoint_group_id, project_id)
    self.get(url, expected_status=http.client.NOT_FOUND)