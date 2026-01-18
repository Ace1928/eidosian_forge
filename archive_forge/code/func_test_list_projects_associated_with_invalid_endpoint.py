import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_list_projects_associated_with_invalid_endpoint(self):
    """GET & HEAD /OS-EP-FILTER/endpoints/{endpoint_id}/projects.

        Invalid endpoint id test case.

        """
    url = '/OS-EP-FILTER/endpoints/%(endpoint_id)s/projects' % {'endpoint_id': uuid.uuid4().hex}
    self.get(url, expected_status=http.client.NOT_FOUND)
    self.head(url, expected_status=http.client.NOT_FOUND)