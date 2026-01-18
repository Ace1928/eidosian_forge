import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
def test_list_endpoints_filtered_by_interface(self):
    """Call ``GET /endpoints?interface={interface}``."""
    ref = self._create_random_endpoint(interface='internal')
    response = self.get('/endpoints?interface=%s' % ref['interface'])
    self.assertValidEndpointListResponse(response, ref=ref)
    for endpoint in response.json['endpoints']:
        self.assertEqual(ref['interface'], endpoint['interface'])