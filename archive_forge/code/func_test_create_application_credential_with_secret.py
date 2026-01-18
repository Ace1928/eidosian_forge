import datetime
from testtools import matchers
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_create_application_credential_with_secret(self):
    with self.test_client() as c:
        secret = 'supersecuresecret'
        roles = [{'id': self.role_id}]
        app_cred_body = self._app_cred_body(roles=roles, secret=secret)
        token = self.get_scoped_token()
        resp = c.post('/v3/users/%s/application_credentials' % self.user_id, json=app_cred_body, expected_status_code=http.client.CREATED, headers={'X-Auth-Token': token})
    self.assertEqual(secret, resp.json['application_credential']['secret'])