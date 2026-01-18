import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_create_endpoint_project_association_with_invalid_endpoint(self):
    """PUT /OS-EP-FILTER/projects/{project_id}/endpoints/{endpoint_id}.

        Invalid endpoint id test case.

        """
    self.put('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': self.default_domain_project_id, 'endpoint_id': uuid.uuid4().hex}, expected_status=http.client.NOT_FOUND)