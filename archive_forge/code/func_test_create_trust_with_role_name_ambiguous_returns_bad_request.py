import datetime
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_create_trust_with_role_name_ambiguous_returns_bad_request(self):
    role_ref = unit.new_role_ref(name=self.role['name'], domain_id=uuid.uuid4().hex)
    self.post('/roles', body={'role': role_ref})
    ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user_id, project_id=self.project_id, role_names=[self.role['name']])
    self.post('/OS-TRUST/trusts', body={'trust': ref}, expected_status=http.client.BAD_REQUEST)