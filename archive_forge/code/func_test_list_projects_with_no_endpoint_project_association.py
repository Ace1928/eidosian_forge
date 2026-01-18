import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_list_projects_with_no_endpoint_project_association(self):
    """GET & HEAD /OS-EP-FILTER/endpoints/{endpoint_id}/projects.

        Valid endpoint id but no endpoint-project associations test case.

        """
    url = '/OS-EP-FILTER/endpoints/%(endpoint_id)s/projects' % {'endpoint_id': self.endpoint_id}
    r = self.get(url, expected_status=http.client.OK)
    self.assertValidProjectListResponse(r, expected_length=0)
    self.head(url, expected_status=http.client.OK)