import datetime
from testtools import matchers
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_create_application_credential_with_application_credential(self):
    with self.test_client() as c:
        roles = [{'id': self.role_id}]
        app_cred_body_1 = self._app_cred_body(roles=roles)
        token = self.get_scoped_token()
        app_cred_1 = c.post('/v3/users/%s/application_credentials' % self.user_id, json=app_cred_body_1, expected_status_code=http.client.CREATED, headers={'X-Auth-Token': token})
        auth_data = self.build_authentication_request(app_cred_id=app_cred_1.json['application_credential']['id'], secret=app_cred_1.json['application_credential']['secret'])
        token_data = self.v3_create_token(auth_data, expected_status=http.client.CREATED)
        app_cred_body_2 = self._app_cred_body(roles=roles)
        token = token_data.headers['x-subject-token']
        c.post('/v3/users/%s/application_credentials' % self.user_id, json=app_cred_body_2, expected_status_code=http.client.FORBIDDEN, headers={'X-Auth-Token': token})