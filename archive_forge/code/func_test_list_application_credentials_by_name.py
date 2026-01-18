import datetime
from testtools import matchers
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_list_application_credentials_by_name(self):
    with self.test_client() as c:
        roles = [{'id': self.role_id}]
        app_cred_body = self._app_cred_body(roles=roles)
        token = self.get_scoped_token()
        name = app_cred_body['application_credential']['name']
        search_path = '/v3/users/%(user_id)s/application_credentials?name=%(name)s' % {'user_id': self.user_id, 'name': name}
        resp = c.get(search_path, expected_status_code=http.client.OK, headers={'X-Auth-Token': token})
        self.assertEqual([], resp.json['application_credentials'])
        resp = c.post('/v3/users/%s/application_credentials' % self.user_id, json=app_cred_body, expected_status_code=http.client.CREATED, headers={'X-Auth-Token': token})
        resp = c.get(search_path, expected_status_code=http.client.OK, headers={'X-Auth-Token': token})
        self.assertEqual(1, len(resp.json['application_credentials']))
        self.assertNotIn('secret', resp.json['application_credentials'][0])
        self.assertNotIn('secret_hash', resp.json['application_credentials'][0])
        app_cred_body['application_credential']['name'] = 'two'
        c.post('/v3/users/%s/application_credentials' % self.user_id, json=app_cred_body, expected_status_code=http.client.CREATED, headers={'X-Auth-Token': token})
        resp = c.get(search_path, expected_status_code=http.client.OK, headers={'X-Auth-Token': token})
        self.assertEqual(1, len(resp.json['application_credentials']))
        self.assertEqual(resp.json['application_credentials'][0]['name'], name)