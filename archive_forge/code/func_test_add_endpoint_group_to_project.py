import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_add_endpoint_group_to_project(self):
    """Create a valid endpoint group and project association."""
    endpoint_group_id = self._create_valid_endpoint_group(self.DEFAULT_ENDPOINT_GROUP_URL, self.DEFAULT_ENDPOINT_GROUP_BODY)
    self._create_endpoint_group_project_association(endpoint_group_id, self.project_id)