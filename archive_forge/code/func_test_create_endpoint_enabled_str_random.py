import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
def test_create_endpoint_enabled_str_random(self):
    """Call ``POST /endpoints`` with enabled: 'puppies'."""
    ref = unit.new_endpoint_ref(service_id=self.service_id, interface='public', region_id=self.region_id, enabled='puppies')
    self.post('/endpoints', body={'endpoint': ref}, expected_status=http.client.BAD_REQUEST)