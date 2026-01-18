import datetime
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_delete_trust_with_application_credential(self):
    ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user_id, project_id=self.project_id, impersonation=False, expires=dict(minutes=1), role_ids=[self.role_id])
    r = self.post('/OS-TRUST/trusts', body={'trust': ref})
    trust = self.assertValidTrustResponse(r, ref)
    app_cred = {'id': uuid.uuid4().hex, 'user_id': self.user_id, 'project_id': self.project_id, 'name': uuid.uuid4().hex, 'roles': [{'id': self.role_id}], 'secret': uuid.uuid4().hex}
    app_cred_api = PROVIDERS.application_credential_api
    app_cred_api.create_application_credential(app_cred)
    auth_data = self.build_authentication_request(app_cred_id=app_cred['id'], secret=app_cred['secret'])
    token_data = self.v3_create_token(auth_data, expected_status=http.client.CREATED)
    self.delete(path='/OS-TRUST/trusts/%(trust_id)s' % {'trust_id': trust['id']}, token=token_data.headers['x-subject-token'], expected_status=http.client.FORBIDDEN)