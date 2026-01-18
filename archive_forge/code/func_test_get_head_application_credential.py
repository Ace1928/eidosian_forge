import datetime
from testtools import matchers
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_get_head_application_credential(self):
    with self.test_client() as c:
        roles = [{'id': self.role_id}]
        app_cred_body = self._app_cred_body(roles=roles)
        token = self.get_scoped_token()
        resp = c.post('/v3/users/%s/application_credentials' % self.user_id, json=app_cred_body, expected_status_code=http.client.CREATED, headers={'X-Auth-Token': token})
        app_cred_id = resp.json['application_credential']['id']
        c.head('/v3%s' % MEMBER_PATH_FMT % {'user_id': self.user_id, 'app_cred_id': app_cred_id}, expected_status_code=http.client.OK, headers={'X-Auth-Token': token})
        expected_response = resp.json
        expected_response['application_credential'].pop('secret')
        resp = c.get('/v3%s' % MEMBER_PATH_FMT % {'user_id': self.user_id, 'app_cred_id': app_cred_id}, expected_status_code=http.client.OK, headers={'X-Auth-Token': token})
        self.assertDictEqual(resp.json, expected_response)