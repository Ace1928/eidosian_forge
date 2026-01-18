import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
def test_list_endpoints_filtered_by_region_id(self):
    """Call ``GET /endpoints?region_id={region_id}``."""
    ref = self._create_random_endpoint()
    response = self.get('/endpoints?region_id=%s' % ref['region_id'])
    self.assertValidEndpointListResponse(response, ref=ref)
    for endpoint in response.json['endpoints']:
        self.assertEqual(ref['region_id'], endpoint['region_id'])