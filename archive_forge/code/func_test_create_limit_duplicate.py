import http.client
import uuid
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_create_limit_duplicate(self):
    ref = unit.new_limit_ref(project_id=self.project_id, service_id=self.service_id, region_id=self.region_id, resource_name='volume')
    self.post('/limits', body={'limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CREATED)
    self.post('/limits', body={'limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CONFLICT)