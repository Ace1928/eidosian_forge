import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
def test_get_head_endpoint(self):
    """Call ``GET & HEAD /endpoints/{endpoint_id}``."""
    resource_url = '/endpoints/%(endpoint_id)s' % {'endpoint_id': self.endpoint_id}
    r = self.get(resource_url)
    self.assertValidEndpointResponse(r, self.endpoint)
    self.head(resource_url, expected_status=http.client.OK)