import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
def test_list_head_endpoints(self):
    """Call ``GET & HEAD /endpoints``."""
    resource_url = '/endpoints'
    r = self.get(resource_url)
    self.assertValidEndpointListResponse(r, ref=self.endpoint)
    self.head(resource_url, expected_status=http.client.OK)