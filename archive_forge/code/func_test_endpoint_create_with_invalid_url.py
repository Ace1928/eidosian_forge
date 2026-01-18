import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
def test_endpoint_create_with_invalid_url(self):
    """Test the invalid cases: substitutions is not exactly right."""
    invalid_urls = ['http://127.0.0.1:8774/v1.1/$(nonexistent)s', 'http://127.0.0.1:8774/v1.1/$(project_id)', 'http://127.0.0.1:8774/v1.1/$(project_id)t', 'http://127.0.0.1:8774/v1.1/$(project_id', 'http://127.0.0.1:8774/v1.1/$(admin_url)d']
    ref = unit.new_endpoint_ref(self.service_id)
    for invalid_url in invalid_urls:
        ref['url'] = invalid_url
        self.post('/endpoints', body={'endpoint': ref}, expected_status=http.client.BAD_REQUEST)