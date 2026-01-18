import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
def test_create_region_with_empty_id(self):
    """Call ``POST /regions`` with an empty ID in the request body."""
    ref = unit.new_region_ref(id='')
    r = self.post('/regions', body={'region': ref})
    self.assertValidRegionResponse(r, ref)
    self.assertNotEmpty(r.result['region'].get('id'))