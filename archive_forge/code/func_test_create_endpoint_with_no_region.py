import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
def test_create_endpoint_with_no_region(self):
    """EndpointV3 allows to creates the endpoint without region."""
    ref = unit.new_endpoint_ref(service_id=self.service_id, region_id=None)
    del ref['region_id']
    self.post('/endpoints', body={'endpoint': ref})