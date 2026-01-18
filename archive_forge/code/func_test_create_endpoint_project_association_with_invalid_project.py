import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_create_endpoint_project_association_with_invalid_project(self):
    """PUT OS-EP-FILTER/projects/{project_id}/endpoints/{endpoint_id}.

        Invalid project id test case.

        """
    self.put('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': uuid.uuid4().hex, 'endpoint_id': self.endpoint_id}, expected_status=http.client.NOT_FOUND)