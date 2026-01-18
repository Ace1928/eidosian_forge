import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_create_endpoint_project_association(self):
    """PUT /OS-EP-FILTER/projects/{project_id}/endpoints/{endpoint_id}.

        Valid endpoint and project id test case.

        """
    self.put(self.default_request_url)