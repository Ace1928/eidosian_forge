import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
def test_endpoint_create_with_valid_url(self):
    """Create endpoint with valid url should be tested,too."""
    valid_url = 'http://127.0.0.1:8774/v1.1/$(project_id)s'
    ref = unit.new_endpoint_ref(self.service_id, interface='public', region_id=self.region_id, url=valid_url)
    self.post('/endpoints', body={'endpoint': ref})