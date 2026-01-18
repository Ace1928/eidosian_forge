import datetime
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_create_trust_with_trustee_as_trustor_returns_forbidden(self):
    ref = unit.new_trust_ref(trustor_user_id=self.trustee_user_id, trustee_user_id=self.user_id, project_id=self.project_id, role_ids=[self.role_id])
    self.post('/OS-TRUST/trusts', body={'trust': ref}, expected_status=http.client.FORBIDDEN)