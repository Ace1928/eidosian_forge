import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_list_projects_associated_with_endpoint_group(self):
    """GET & HEAD /OS-EP-FILTER/endpoint_groups/{endpoint_group}/projects.

        Valid endpoint group test case.

        """
    endpoint_group_id = self._create_valid_endpoint_group(self.DEFAULT_ENDPOINT_GROUP_URL, self.DEFAULT_ENDPOINT_GROUP_BODY)
    self._create_endpoint_group_project_association(endpoint_group_id, self.project_id)
    url = '/OS-EP-FILTER/endpoint_groups/%(endpoint_group_id)s/projects' % {'endpoint_group_id': endpoint_group_id}
    self.get(url, expected_status=http.client.OK)
    self.head(url, expected_status=http.client.OK)