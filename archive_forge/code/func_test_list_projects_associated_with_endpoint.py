import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_list_projects_associated_with_endpoint(self):
    """GET & HEAD /OS-EP-FILTER/endpoints/{endpoint_id}/projects.

        Valid endpoint-project association test case.

        """
    self.put(self.default_request_url)
    resource_url = '/OS-EP-FILTER/endpoints/%(endpoint_id)s/projects' % {'endpoint_id': self.endpoint_id}
    r = self.get(resource_url, expected_status=http.client.OK)
    self.assertValidProjectListResponse(r, self.default_domain_project, resource_url=resource_url)
    self.head(resource_url, expected_status=http.client.OK)