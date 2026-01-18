import datetime
from testtools import matchers
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_create_application_credential_with_trust(self):
    second_role = unit.new_role_ref(name='reader')
    PROVIDERS.role_api.create_role(second_role['id'], second_role)
    PROVIDERS.assignment_api.add_role_to_user_and_project(self.user_id, self.project_id, second_role['id'])
    with self.test_client() as c:
        pw_token = self.get_scoped_token()
        trust_ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.user_id, project_id=self.project_id, role_ids=[second_role['id']])
        resp = c.post('/v3/OS-TRUST/trusts', headers={'X-Auth-Token': pw_token}, json={'trust': trust_ref})
        trust_id = resp.json['trust']['id']
        trust_auth = self.build_authentication_request(user_id=self.user_id, password=self.user['password'], trust_id=trust_id)
        trust_token = self.v3_create_token(trust_auth).headers['X-Subject-Token']
        app_cred = self._app_cred_body(roles=[{'id': self.role_id}])
        c.post('/v3/users/%s/application_credentials' % self.user_id, headers={'X-Auth-Token': trust_token}, json=app_cred, expected_status_code=http.client.BAD_REQUEST)